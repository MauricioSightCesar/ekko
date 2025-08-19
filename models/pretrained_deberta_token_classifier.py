"""DeBERTa-based NER using Hugging Face Transformers.

Mirrors the interface of the Flair-based model but uses the common
Transformers token-classification pipeline. Includes sequence chunking
and evaluator-format conversion using existing utils.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import time

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline as hf_pipeline,
)

# No Flair-specific utilities here; we expose raw HF pipeline outputs.


class PretrainedDebertaTokenClassifierModel:
    """Token-classification NER with DeBERTa via Transformers.

    Default model_name is a base DeBERTa checkpoint. If it lacks a
    token-classification head, predictions will be poor. The class will
    attempt to proceed but you may prefer a fine-tuned NER checkpoint.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "microsoft/deberta-v3-base",
        aggregation_strategy: str = "simple",
        logger: Optional[Any] = None,
    ) -> None:
        """Initialize DeBERTa token-classifier.

        Args:
            device: 'cpu' or 'cuda'.
            model_name: Hugging Face model id.
            aggregation_strategy: Token aggregation for pipeline results ('simple', 'first', 'max', 'average').
            logger: Optional logger instance.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        self.logger = logger

        # Load tokenizer and model (most common approach with Transformers)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model/tokenizer '{self.model_name}': {e}")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.model = self.model.to(self.device)

        self.model.eval()
        self._log_info(f"Loaded Transformers model: {self.model_name} on {self.device}")

        # Build pipeline (handles batching under the hood for lists of texts)
        pipeline_device = 0 if self.device.type == "cuda" else -1
        self.pipe = hf_pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=self.aggregation_strategy,
            device=pipeline_device,
        )

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)

    # --- Chunking helpers (reused design) ---------------------------------
    def _chunk_long_sequences(self, test_data: pd.DataFrame, max_length: int = 384) -> pd.DataFrame:
        """Chunk long sequences into smaller pieces to prevent memory issues.

        Args:
            test_data: DataFrame with 'source_text' and 'span_labels' columns
            max_length: Maximum number of whitespace-tokenized tokens per chunk
        """
        chunked_rows: List[pd.Series] = []
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
        original_tokens: List[str],
    ) -> List[List]:
        """Adjust span labels for a text chunk by filtering and repositioning spans."""
        if not span_labels:
            return []

        # Calculate character positions for token boundaries
        char_positions: List[int] = []
        current_pos = 0
        for token in original_tokens:
            char_positions.append(current_pos)
            current_pos += len(token) + 1  # +1 for space
        char_positions.append(current_pos - 1)

        chunk_start_char = char_positions[start_token_idx]
        chunk_end_char = char_positions[end_token_idx] if end_token_idx < len(char_positions) else char_positions[-1]

        adjusted_spans: List[List] = []
        for span in span_labels:
            span_start, span_end, span_label = span[0], span[1], span[2]
            if span_start < chunk_end_char and span_end > chunk_start_char:
                adjusted_start = max(0, span_start - chunk_start_char)
                adjusted_end = min(chunk_end_char - chunk_start_char, span_end - chunk_start_char)
                if adjusted_end > adjusted_start:
                    adjusted_spans.append([adjusted_start, adjusted_end, span_label])
        return adjusted_spans

    # --- Evaluation --------------------------------------------------------
    def evaluate(self, test_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[List[List[Tuple[str, float]]], List[List[str]]]:
        """Return token-level predictions and ground truth per whitespace token.

        Args:
            test_data: DataFrame with 'source_text'.
            config: Unused; kept for API parity.

        Returns:
            Tuple (
                predictions: List of samples, each a list of (label, score) per whitespace token,
                ground_truth: List of samples, each a list of label strings per whitespace token
            ).
        """
        self._log_info(f"Evaluating model on {len(test_data)} samples (token-level)")
        # Chunk long sequences similarly to GLiNER (by 384 whitespace tokens)
        chunked = self._chunk_long_sequences(test_data)
        texts: List[str] = chunked.get('source_text', pd.Series(dtype=str)).astype(str).tolist()
        gt_spans_list: List[Any] = (
            chunked.get('span_labels', pd.Series(dtype=object)).tolist()
            if 'span_labels' in chunked else
            [[] for _ in texts]
        )
        if not texts:
            return [], []

        # Minimal label rename to resemble examples; otherwise keep as-is
        label_map = {
            "PER": "PERSON_NAME",
            "PERSON": "PERSON_NAME",
            "ORG": "ORGANIZATION",
            "ORGANIZATION": "ORGANIZATION",
            "LOC": "LOCATION"
        }

        def whitespace_tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
            tokens = text.split()
            positions: List[Tuple[int, int]] = []
            cursor = 0
            for tok in tokens:
                # skip whitespace
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                start = text.find(tok, cursor)
                if start == -1:
                    start = cursor
                end = start + len(tok)
                positions.append((start, end))
                cursor = end
            return tokens, positions

        predictions: List[List[Tuple[str, float]]] = []
        ground_truth: List[List[str]] = []

        eval_start = time.time()
        batch_size = self._get_optimal_batch_size(len(texts))

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                batch_outputs = self.pipe(batch_texts)
            except Exception as e:
                self._log_info(f"Batch prediction failed: {e}")
                # Fill defaults to maintain shape
                for text in batch_texts:
                    tok_list, _ = whitespace_tokenize_with_offsets(text)
                    predictions.append([("0", 0.1) for _ in tok_list])
                    ground_truth.append(["0" for _ in tok_list])
                continue

            # Normalize output to list-per-input
            if not isinstance(batch_outputs, list) or (batch_texts and not isinstance(batch_outputs[0], list)):
                batch_outputs = [batch_outputs] if isinstance(batch_outputs, list) else [[batch_outputs]]

            for idx_in_batch, (text, ents) in enumerate(zip(batch_texts, batch_outputs)):
                tok_list, tok_positions = whitespace_tokenize_with_offsets(text)
                token_preds: List[Tuple[str, float]] = [("0", 0.1) for _ in tok_list]

                for ent in ents or []:
                    raw_label = ent.get("entity_group") or ent.get("entity") or "0"
                    mapped_label = label_map.get(str(raw_label), "0")
                    start = int(ent.get("start", 0))
                    end = int(ent.get("end", start))
                    score = float(ent.get("score", 0.9))
                    score = max(0.1, score)

                    for idx, (ts, te) in enumerate(tok_positions):
                        if ts < end and te > start:  # overlap
                            token_preds[idx] = (mapped_label, score)

                predictions.append(token_preds)
                # Build ground-truth labels per token from provided span_labels
                gt_labels = ["0" for _ in tok_list]
                global_idx = i + idx_in_batch
                sample_spans = gt_spans_list[global_idx] if global_idx < len(gt_spans_list) else []
                if isinstance(sample_spans, (list, tuple)):
                    for span in sample_spans:
                        if isinstance(span, dict):
                            s = int(span.get("start", 0))
                            e = int(span.get("end", 0))
                            lbl = str(span.get("label", "0"))
                        elif isinstance(span, (list, tuple)) and len(span) >= 3:
                            s = int(span[0])
                            e = int(span[1])
                            lbl = str(span[2])
                        else:
                            continue
                        for t_idx, (ts, te) in enumerate(tok_positions):
                            if ts < e and te > s:
                                gt_labels[t_idx] = lbl
                ground_truth.append(gt_labels)

        self._log_info(f"Evaluation completed in {time.time() - eval_start:.2f}s")
        return predictions, ground_truth

    # --- Utilities ---------------------------------------------------------
    def _get_optimal_batch_size(self, total_samples: int) -> int:
        """Choose a conservative batch size based on device and sample count."""
        if self.device.type != "cuda":
            return min(16, max(1, total_samples))
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            if gpu_mem_gb >= 24:
                return 64
            if gpu_mem_gb >= 12:
                return 32
            return 16
        except Exception:
            return 16

    def to(self, device):
        """Move model to device for API parity."""
        self.device = device
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        return self

    def eval(self):
        """Set model to eval mode for API parity."""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        return self

    def train(self):
        """Set model to train mode for API parity."""
        if hasattr(self.model, 'train'):
            self.model.train()
        return self

    def state_dict(self):
        """Return state dict placeholder for parity (pretrained models)."""
        if hasattr(self.model, 'state_dict'):
            return self.model.state_dict()
        return {"pretrained_model": self.model_name}

    def load_state_dict(self, state_dict):
        """No-op for hub-loaded models; parity with other classes."""
        return None
