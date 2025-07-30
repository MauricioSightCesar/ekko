import torch
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from typing import List, Dict, Tuple, Optional, Any, Union
import time
import multiprocessing as mp
import concurrent.futures
from functools import partial

# Import NER processing utilities
from utils.ner_processing_utils import (
    convert_bio_to_spans_worker_complete,
    spans_to_evaluator_format,
)


class PretrainedLSTMCRFFlairModel:
    """
    LSTM-CRF model using pre-trained Flair OntoNotes model for named entity recognition.
    
    Handles long sequences through chunking and provides parallel processing
    for efficient batch prediction and evaluation.
    """

    def __init__(
        self, 
        device: str = "cuda", 
        model_name: str = "flair/ner-english-ontonotes", 
        logger: Optional[Any] = None
    ) -> None:
        """
        Initialize pre-trained LSTM-CRF model.

        Args:
            device: Device to use ('cpu' or 'cuda')
            model_name: Name of the pre-trained Flair model
            logger: Logger instance for logging messages
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_name = model_name
        self.logger = logger
        self.tagger = SequenceTagger.load(model_name)
        
        self._log_info(f"Using LSTM-CRF with device: {self.device}")
        
        # GPU optimization settings
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.tagger = self.tagger.to(self.device)
            self.use_mixed_precision = True
            self._log_info("GPU optimizations enabled")
        else:
            self.use_mixed_precision = False
            
        # Set model to evaluation mode
        self.tagger.eval()
        
        # Initialize CPU cores for parallel processing
        self.num_cpu_cores = min(8, mp.cpu_count())
        self._log_info(f"CPU cores available for parallel processing: {self.num_cpu_cores}")
        
        self._log_info("Pre-trained model loaded successfully")

    def _log_info(self, message: str) -> None:
        """Log info messages using the configured logger."""
        if self.logger:
            self.logger.info(message)

    def _get_optimal_mini_batch_size(self, num_sentences: int) -> int:
        """
        Calculate optimal mini-batch size for Flair prediction.
        
        Args:
            num_sentences: Total number of sentences to process
            
        Returns:
            Optimal mini-batch size
        """
        if self.device.type == "cuda":
            # GPU: Use larger batches for efficiency
            if num_sentences >= 1000:
                return 64
            elif num_sentences >= 100:
                return 32
            else:
                return 16
        else:
            # CPU: Use smaller batches to avoid memory issues
            if num_sentences >= 1000:
                return 16
            elif num_sentences >= 100:
                return 8
            else:
                return 4

    def _chunk_long_sequences(self, test_data: pd.DataFrame, max_length: int = 384) -> pd.DataFrame:
        """
        Chunk long sequences into smaller pieces to prevent memory issues.
        
        Args:
            test_data: DataFrame with 'source_text' and 'span_labels' columns
            max_length: Maximum number of tokens per chunk
            
        Returns:
            DataFrame with chunked sequences (may have more rows than input)
        """
        chunked_rows = []
        chunk_count = 0
        
        for idx, row in test_data.iterrows():
            source_text = row['source_text']
            span_labels = row['span_labels']
            tokens = source_text.split()
            text_size = len(tokens)
            
            if text_size > max_length:
                num_chunks = (text_size // max_length) + 1
                self._log_info(f"Sample {idx}: {text_size} tokens -> {num_chunks} chunks")
                
                for j in range(num_chunks):
                    start_idx = j * max_length
                    end_idx = min(text_size, (j + 1) * max_length)
                    
                    chunk_tokens = tokens[start_idx:end_idx]
                    chunk_text = ' '.join(chunk_tokens)
                    chunk_labels = self._adjust_spans_for_chunk(span_labels, start_idx, end_idx, tokens)
                    
                    chunked_row = row.copy()
                    chunked_row['source_text'] = chunk_text
                    chunked_row['span_labels'] = chunk_labels
                    chunked_rows.append(chunked_row)
                    chunk_count += 1
            else:
                chunked_rows.append(row)
        
        if chunk_count > 0:
            self._log_info(f"Created {chunk_count} chunks from long sequences")
        
        return pd.DataFrame(chunked_rows)

    def _adjust_spans_for_chunk(
        self, 
        span_labels: List[List], 
        start_token_idx: int, 
        end_token_idx: int, 
        original_tokens: List[str]
    ) -> List[List]:
        """
        Adjust span labels for a text chunk by filtering and repositioning spans.
        
        Args:
            span_labels: Original span labels (list of [start_char, end_char, label])
            start_token_idx: Start token index of the chunk
            end_token_idx: End token index of the chunk
            original_tokens: Original tokenized text
            
        Returns:
            Adjusted span labels for the chunk
        """
        if not span_labels:
            return []
            
        # Calculate character positions for token boundaries
        char_positions = []
        current_pos = 0
        for token in original_tokens:
            char_positions.append(current_pos)
            current_pos += len(token) + 1  # +1 for space
        char_positions.append(current_pos - 1)  # End position
        
        chunk_start_char = char_positions[start_token_idx]
        chunk_end_char = char_positions[end_token_idx] if end_token_idx < len(char_positions) else char_positions[-1]
        
        adjusted_spans = []
        
        for span in span_labels:
            span_start, span_end, span_label = span[0], span[1], span[2]
            
            # Check if span overlaps with chunk
            if span_start < chunk_end_char and span_end > chunk_start_char:
                adjusted_start = max(0, span_start - chunk_start_char)
                adjusted_end = min(chunk_end_char - chunk_start_char, span_end - chunk_start_char)
                
                if adjusted_end > adjusted_start:
                    adjusted_spans.append([adjusted_start, adjusted_end, span_label])
        
        return adjusted_spans

    def evaluate(
        self, 
        test_data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Tuple[List, List]:
        """
        Evaluate the model on test data using batch prediction.

        Args:
            test_data: DataFrame with 'source_text' and 'span_labels' columns
            config: Configuration dictionary containing dataset and feature settings

        Returns:
            Tuple of (predictions, ground_truth) in token-level format
        """
        self._log_info(f"Evaluating model on {len(test_data)} samples")

        # Extract configuration
        entity_types = config.get("feature", {}).get("labels", [])
        dataset_name = config.get("dataset", {}).get("name")
        eval_start = time.time()
        
        # Stage 1: Chunking
        chunking_start = time.time()
        self._log_info("Applying sequence chunking (max 384 tokens)")
        chunked_test_data = self._chunk_long_sequences(test_data)
        chunking_time = time.time() - chunking_start
        self._log_info(f"Chunking completed: {len(test_data)} -> {len(chunked_test_data)} samples")
        
        # Stage 2: Sentence Creation
        sentence_creation_start = time.time()
        self._log_info("Creating Flair sentences")
        all_prepared_data = self._create_all_sentences_parallel(chunked_test_data, entity_types, dataset_name)
        sentence_creation_time = time.time() - sentence_creation_start
        self._log_info(f"Sentence creation completed: {len(all_prepared_data)} valid sentences")
        
        if not all_prepared_data:
            return [], []

        # Stage 3: Batch Prediction
        gpu_start = time.time()
        self._log_info("Running batch prediction")
        all_sentences = [data['sentence'] for data in all_prepared_data]
        mini_batch_size = self._get_optimal_mini_batch_size(len(all_sentences))
        
        try:
            with torch.no_grad():
                self.tagger.predict(all_sentences, mini_batch_size=mini_batch_size)
        except Exception as e:
            self._log_info(f"Batch prediction failed: {e}")
            return [], []

        gpu_time = time.time() - gpu_start
        self._log_info(f"Batch prediction completed in {gpu_time:.2f}s")
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Stage 4: CPU Processing
        cpu_start = time.time()
        self._log_info("Processing predictions")
        
        cpu_processing_data = []
        for data in all_prepared_data:
            cpu_processing_data.append({
                'sentence': data['sentence'],
                'span_labels': data['span_labels'], 
                'source_text': data['source_text'],
                'entity_types': data['entity_types'],
                'dataset_name': data.get('dataset_name'),
                'sample_idx': data['sample_idx']
            })

        predictions, ground_truth = self._process_all_cpu_parallel(cpu_processing_data)
        cpu_time = time.time() - cpu_start
        
        eval_time = time.time() - eval_start
        self._log_info(f"Evaluation completed in {eval_time:.2f}s")
        self._log_info(f"Processed {len(predictions)} samples")
        
        return predictions, ground_truth

    def _create_all_sentences_parallel(
        self, test_data: pd.DataFrame, entity_types: List[str], dataset_name: str = None
    ) -> List[Dict]:
        """
        Create all sentences in parallel - ELIMINATES pandas bottlenecks
        
        Args:
            test_data: DataFrame with source text and span labels
            entity_types: List of target entity types
        
        Returns:
            List of prepared data with pre-created sentences
        """
        # OPTIMIZATION 1: Extract ALL data from pandas at once (no iterrows!)
        source_texts = test_data["source_text"].tolist()
        span_labels_list = test_data["span_labels"].tolist()
        
        # OPTIMIZATION 2: Filter valid samples in vectorized way
        valid_mask = [bool(text and str(text).strip()) for text in source_texts]
        
        # Create list of valid data for parallel processing
        valid_data = [
            (source_texts[i], span_labels_list[i], entity_types, dataset_name, i)
            for i, is_valid in enumerate(valid_mask) if is_valid
        ]
        
        if not valid_data:
            return []

        self._log_info(f"   📊 Found {len(valid_data)} valid samples out of {len(test_data)}")

        def create_sentence_data(text_data):
            """Worker function to create a single sentence with metadata"""
            text, spans, entity_types, dataset_name, sample_idx = text_data
            try:
                sentence = Sentence(text)
                if sentence.tokens:  # Only return if tokenization succeeded
                    return {
                        'sentence': sentence,
                        'span_labels': spans,
                        'source_text': text,
                        'entity_types': entity_types,
                        'dataset_name': dataset_name,
                        'sample_idx': sample_idx
                    }
                else:
                    return None
            except Exception:
                return None

        # OPTIMIZATION 3: Parallel sentence creation with optimal worker count
        prepared_data = []
        max_workers = min(8, len(valid_data), mp.cpu_count())
        
        self._log_info(f"   🔄 Using {max_workers} workers for parallel sentence creation...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_data = {
                executor.submit(create_sentence_data, text_data): text_data 
                for text_data in valid_data
            }
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_data):
                result = future.result()
                if result is not None:
                    prepared_data.append(result)
                
                completed += 1
                # Progress reporting for large datasets
                if completed % 100 == 0:
                    self._log_info(f"   📝 Created {completed}/{len(valid_data)} sentences...")

        self._log_info(f"   ✅ Successfully created {len(prepared_data)} sentences")
        return prepared_data

    def _get_optimal_batch_size(self, total_samples: int) -> int:
        """
        Determine optimal batch size based on GPU memory and dataset size

        Args:
            total_samples: Total number of samples to process

        Returns:
            Optimal batch size
        """
        if self.device.type != "cuda":
            return min(32, total_samples)  # Conservative batch size for CPU

        # Get available GPU memory
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(
                self.device
            ).total_memory / (1024**3)
            available_memory_gb = gpu_memory_gb * 0.8  # Use 80% of available memory

            # Estimate memory per sample (rough estimate based on text processing)
            estimated_memory_per_sample_mb = 50  # MB per sample
            max_batch_from_memory = int(
                (available_memory_gb * 1024) / estimated_memory_per_sample_mb
            )

            # Choose batch size based on memory and dataset characteristics
            if total_samples < 100:
                batch_size = min(16, total_samples)
            elif total_samples < 1000:
                batch_size = min(32, max_batch_from_memory, total_samples)
            else:
                batch_size = min(64, max_batch_from_memory)

            self._log_info(
                f"💾 GPU Memory: {gpu_memory_gb:.1f}GB total, using batch size: {batch_size}"
            )
            return max(1, batch_size)

        except Exception as e:
            self._log_info(f"⚠️  Could not determine GPU memory, using default batch size: {e}")
            return 32

    def _get_optimal_mini_batch_size(self, total_sentences: int) -> int:
        """
        Determine optimal mini_batch_size for Flair's predict method
        
        Args:
            total_sentences: Total number of sentences to process
            
        Returns:
            Optimal mini_batch_size for Flair
        """
        if self.device.type != "cuda":
            return min(8, total_sentences)  # Conservative for CPU
        
        # Get available GPU memory
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(
                self.device
            ).total_memory / (1024**3)
            
            # Determine mini_batch_size based on GPU memory and dataset size
            if gpu_memory_gb >= 24:  # High-end GPU
                if total_sentences < 100:
                    mini_batch_size = min(16, total_sentences)
                elif total_sentences < 1000:
                    mini_batch_size = min(32, total_sentences)
                else:
                    mini_batch_size = 64
            elif gpu_memory_gb >= 8:  # Mid-range GPU
                if total_sentences < 100:
                    mini_batch_size = min(8, total_sentences)
                elif total_sentences < 1000:
                    mini_batch_size = min(16, total_sentences)
                else:
                    mini_batch_size = 32
            else:  # Low-end GPU
                mini_batch_size = min(8, total_sentences)
            
            self._log_info(f"💾 GPU Memory: {gpu_memory_gb:.1f}GB, optimal mini_batch_size: {mini_batch_size}")
            return max(1, mini_batch_size)
            
        except Exception as e:
            self._log_info(f"⚠️  Could not determine GPU memory, using default mini_batch_size: {e}")
            return 16

    def _process_all_cpu_parallel(self, all_cpu_processing_data: List[Dict]) -> Tuple[List, List]:
        """
        STAGES 2 & 3: Pre-parallel processing + Parallel CPU processing
        
        Stage 2: Extract data from Flair objects (main process, one-time)
        Stage 3: Heavy CPU processing in parallel workers
        
        Args:
            all_cpu_processing_data: List of dictionaries with raw Flair objects

        Returns:
            Tuple of (batch_predictions, batch_ground_truth) in evaluator format
        """
        batch_predictions = []
        batch_ground_truth = []

        if not all_cpu_processing_data:
            return batch_predictions, batch_ground_truth

        # STAGE 2: PRE-PARALLEL PROCESSING - Extract data from Flair objects
        self._log_info(f"🔧 Extracting data from {len(all_cpu_processing_data)} Flair objects...")
        prep_start = time.time()
        
        clean_processing_data = []
        for data in all_cpu_processing_data:
            try:
                sentence = data['sentence']
                
                # Extract tokens as strings (one-time, in main process)
                tokens = [str(token) for token in sentence.tokens]
                
                # Extract raw entity data from Flair predictions
                raw_entities = []
                for entity in sentence.get_spans("ner"):
                    raw_entities.append({
                        'text': entity.text,
                        'tag': entity.tag,
                        'score': float(entity.score),
                        'start_position': entity.start_position,
                        'end_position': entity.end_position
                    })
                
                # Store clean, picklable data for workers
                clean_processing_data.append({
                    'tokens': tokens,
                    'raw_entities': raw_entities,
                    'span_labels': data['span_labels'],
                    'source_text': data['source_text'],
                    'entity_types': data['entity_types'],
                    'dataset_name': data.get('dataset_name'),
                    'sample_idx': data['sample_idx']
                })
                
            except Exception as e:
                self._log_info(f"⚠️  Error extracting data from sample {data.get('sample_idx', 'unknown')}: {e}")
                continue
        
        prep_time = time.time() - prep_start
        self._log_info(f"✅ Data extraction completed in {prep_time:.3f}s")

        # STAGE 3: PARALLEL CPU PROCESSING - Heavy algorithmic work
        parallel_start = time.time()
        
        if len(clean_processing_data) > 1 and self.num_cpu_cores > 1:
            self._log_info(f"🔄 Processing {len(clean_processing_data)} samples in parallel using {self.num_cpu_cores} cores")
            
            with mp.Pool(processes=self.num_cpu_cores) as pool:
                try:
                    # Use the complete worker function that does all heavy processing
                    results = pool.map(convert_bio_to_spans_worker_complete, clean_processing_data)
                    
                    # Process results
                    for i, result in enumerate(results):
                        if result[0] is not None:  # Check if processing succeeded
                            pred_spans, true_spans = result
                            
                            # Convert spans to evaluator format
                            num_tokens = len(clean_processing_data[i]['tokens'])
                            pred_tokens, true_tokens = spans_to_evaluator_format(
                                pred_spans, true_spans, num_tokens
                            )
                            
                            batch_predictions.append(pred_tokens)
                            batch_ground_truth.append(true_tokens)
                        else:
                            # Error case - log the error message
                            self._log_info(f"⚠️  {result[1]}")
                                
                except Exception as e:
                    self._log_info(f"⚠️  Parallel processing failed, falling back to sequential: {e}")
                    # Fallback to sequential processing
                    for data in clean_processing_data:
                        try:
                            result = convert_bio_to_spans_worker_complete(data)
                            if result[0] is not None:
                                pred_spans, true_spans = result
                                num_tokens = len(data['tokens'])
                                pred_tokens, true_tokens = spans_to_evaluator_format(
                                    pred_spans, true_spans, num_tokens
                                )
                                batch_predictions.append(pred_tokens)
                                batch_ground_truth.append(true_tokens)
                            else:
                                self._log_info(f"⚠️  {result[1]}")
                        except Exception as se:
                            self._log_info(f"⚠️  Error in sequential fallback for sample {data.get('sample_idx', 'unknown')}: {se}")
        else:
            # Sequential processing for small datasets or single core
            self._log_info(f"🔄 Processing {len(clean_processing_data)} samples sequentially")
            for data in clean_processing_data:
                try:
                    result = convert_bio_to_spans_worker_complete(data)
                    if result[0] is not None:
                        pred_spans, true_spans = result
                        num_tokens = len(data['tokens'])
                        pred_tokens, true_tokens = spans_to_evaluator_format(
                            pred_spans, true_spans, num_tokens
                        )
                        batch_predictions.append(pred_tokens)
                        batch_ground_truth.append(true_tokens)
                    else:
                        self._log_info(f"⚠️  {result[1]}")
                except Exception as e:
                    self._log_info(f"⚠️  Error in sequential processing for sample {data.get('sample_idx', 'unknown')}: {e}")
        
        parallel_time = time.time() - parallel_start
        total_time = prep_time + parallel_time
        self._log_info(f"✅ CPU processing completed in {total_time:.3f}s (prep: {prep_time:.3f}s, parallel: {parallel_time:.3f}s)")
        self._log_info(f"   Processing rate: {len(clean_processing_data)/total_time:.1f} samples/sec")

        return batch_predictions, batch_ground_truth

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if self.device.type == "cuda":
            try:
                return torch.cuda.memory_allocated(self.device) / (1024**3)
            except:
                return 0.0
        return 0.0

    def to(self, device):
        """Move model to device (for compatibility with existing code)"""
        self.device = device
        if hasattr(self.tagger, "to"):
            self.tagger = self.tagger.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode (for compatibility)"""
        if hasattr(self.tagger, "eval"):
            self.tagger.eval()
        return self

    def train(self):
        """Set model to training mode (for compatibility)"""
        if hasattr(self.tagger, "train"):
            self.tagger.train()
        return self

    def state_dict(self):
        """Return model state dict (for compatibility with saving)"""
        if hasattr(self.tagger, "state_dict"):
            return self.tagger.state_dict()
        else:
            # Return a placeholder state dict for pre-trained models
            return {"pretrained_model": self.model_name}

    def load_state_dict(self, state_dict):
        """Load model state dict (for compatibility)"""
        # For pre-trained models, we don't actually load anything
        # since the model is loaded from the hub
        pass
