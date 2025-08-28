"""
NER Processing Utilities

Utilities for converting between BIO tags, spans, and token-level representations,
plus helpers for evaluator-format conversions and dataset-aware entity mappings.

Design notes:
- Stateless, multiprocessing-friendly functions.
- Minimal logging: only warnings on exceptional/fallback paths. No prints.
"""
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import os
import json

# Public type aliases
Span = Dict[str, Any]

# Module-level logger (no configuration here)
logger = logging.getLogger(__name__)


def bio_to_spans(
    bio_labels: List[str],
    scores: List[float],
    tokens: List[str],
    source_text: str,
) -> List[Span]:
    """
    Convert BIO labels to contiguous character spans.

    Args:
        bio_labels: BIO labels for each token (e.g., "B-PER", "I-PER", "O").
        scores: Confidence score per token (same length as bio_labels).
        tokens: Token strings corresponding to the labels.
        source_text: Original text for character offsets.

    Returns:
        List of spans, each span as a dict with keys: start, end, label, score, text.
    """
    spans: List[Span] = []
    # Track current entity with explicit fields to avoid Optional subscripting
    current_label: Optional[str] = None
    current_start: int = 0
    current_end: int = 0
    current_score: float = 1.0
    current_text_parts: List[str] = []
    
    # Extract raw token text and create position mapping
    raw_tokens = []
    for token in tokens:
        if isinstance(token, str):
            if token.startswith("Token[") and ']: "' in token:
                start = token.find(']: "') + 4
                end = token.rfind('"')
                raw_tokens.append(token[start:end])
            else:
                raw_tokens.append(token)
        else:
            raw_tokens.append(str(token))
    
    # Create token position mapping (character offsets per token)
    token_positions = []
    current_char_pos = 0
    
    for token_text in raw_tokens:
        # Skip whitespace
        while (
            current_char_pos < len(source_text)
            and source_text[current_char_pos].isspace()
        ):
            current_char_pos += 1
        
        # Find token position
        token_start = source_text.find(token_text, current_char_pos)
        if token_start != -1:
            token_end = token_start + len(token_text)
            token_positions.append((token_start, token_end))
            current_char_pos = token_end
        else:
            # Fallback: approximate position
            token_positions.append(
                (current_char_pos, current_char_pos + len(token_text))
            )
            current_char_pos += len(token_text)
    
    # Process BIO labels to create spans
    for i, label in enumerate(bio_labels):
        if i >= len(token_positions) or i >= len(scores):
            continue
            
        if label.startswith('B-'):
            # Start of new entity - save previous if exists
            if current_label is not None:
                spans.append({
                    'start': current_start,
                    'end': current_end,
                    'label': current_label,
                    'score': current_score,
                    'text': ' '.join(current_text_parts),
                })
            
            # Start new entity
            entity_type = label[2:]
            start_pos, end_pos = token_positions[i]
            current_label = entity_type
            current_start = start_pos
            current_end = end_pos
            current_score = scores[i]
            current_text_parts = [raw_tokens[i]]
            
        elif label.startswith('I-') and current_label is not None:
            # Continue current entity
            entity_type = label[2:]
            if entity_type == current_label:
                # Extend current entity
                _, end_pos = token_positions[i]
                current_end = end_pos
                current_text_parts.append(raw_tokens[i])
                current_score = min(current_score, scores[i])
            else:
                # Type mismatch - save current and start new
                spans.append({
                    'start': current_start,
                    'end': current_end,
                    'label': current_label,
                    'score': current_score,
                    'text': ' '.join(current_text_parts),
                })
                start_pos, end_pos = token_positions[i]
                current_label = entity_type
                current_start = start_pos
                current_end = end_pos
                current_score = scores[i]
                current_text_parts = [raw_tokens[i]]
        
        elif label in ['O', '0']:
            # End current entity
            if current_label is not None:
                spans.append({
                    'start': current_start,
                    'end': current_end,
                    'label': current_label,
                    'score': current_score,
                    'text': ' '.join(current_text_parts),
                })
                current_label = None
    
    # Add final entity if exists
    if current_label is not None:
        spans.append({
            'start': current_start,
            'end': current_end,
            'label': current_label,
            'score': current_score,
            'text': ' '.join(current_text_parts),
        })
    
    return spans


def spans_to_bio_labels(
    spans: List[Span],
    num_tokens: int,
    token_positions: List[Tuple[int, int]],
) -> List[str]:
    """
    Convert spans to BIO labels using token positions.

    Args:
        spans: List of spans with character offsets.
        num_tokens: Number of tokens to label.
        token_positions: Character offsets per token (start, end).

    Returns:
        BIO labels per token index.
    """
    bio_labels = ["0"] * num_tokens
    
    for span in spans:
        span_start = span['start']
        span_end = span['end']
        entity_type = span['label']
        
        first_token = True
        for token_idx, (token_start, token_end) in enumerate(token_positions):
            # Check if token overlaps with span
            if token_start < span_end and token_end > span_start:
                if first_token:
                    bio_labels[token_idx] = f"B-{entity_type}"
                    first_token = False
                else:
                    bio_labels[token_idx] = f"I-{entity_type}"
    
    return bio_labels


def convert_bio_to_spans_worker_complete(
    data: Dict[str, Any]
) -> Tuple[Optional[List[Span]], Optional[Union[List[Span], str]]]:
    """
    Complete worker that does ALL CPU processing from raw Flair data
    
    This function performs all the heavy CPU work that was previously done in the GPU stage:
    1. Create entity mapping
    2. Convert raw entities to BIO tags (heavy token position computation) 
    3. Align ground truth spans to BIO tags
    4. Convert both to spans
    
    Args:
        data: Dictionary containing:
            - tokens: List of token strings (already extracted)
            - raw_entities: Raw entity data from Flair predictions
            - entity_types: Target entity types from config
            - span_labels: Ground truth spans
            - source_text: Original text
            - sample_idx: Sample index for error reporting
    
    Returns:
        Tuple of (predicted_spans, true_spans) or (None, error_message)
    """
    try:
        tokens = data['tokens']
        raw_entities = data['raw_entities']
        entity_types = data['entity_types']
        dataset_name = data.get('dataset_name')  # Get dataset name if available
        span_labels = data['span_labels']
        source_text = data['source_text']
        
        # STEP 1: Create entity mapping with dataset-specific filtering
        entity_mapping = create_entity_mapping(entity_types, dataset_name)
        
        # STEP 2: Convert raw entities to BIO tags (heavy CPU work moved from GPU)
        predicted_bio_tags, pred_scores = raw_entities_to_bio_tags(
            raw_entities, tokens, source_text, entity_mapping
        )
        
        # STEP 3: Align ground truth spans to BIO tags (moved from GPU stage)
        true_bio_tags = align_spans_to_bio_tags(span_labels, tokens, source_text)
        
        # STEP 4: Convert both to spans
        pred_spans = bio_to_spans(predicted_bio_tags, pred_scores, tokens, source_text)
        true_spans = bio_to_spans(true_bio_tags, [0.5] * len(true_bio_tags), tokens, source_text)
        
        return (pred_spans, true_spans)
        
    except (KeyError, TypeError, ValueError) as e:
        return (None, f"Error in worker for sample {data.get('sample_idx', 'unknown')}: {e}")


def raw_entities_to_bio_tags(
    raw_entities: List[Dict[str, Any]],
    tokens: List[str],
    source_text: str,
    entity_mapping: Dict[str, str],
) -> Tuple[List[str], List[float]]:
    """
    Convert raw Flair entities to BIO tags - the heavy CPU work from _extract_predictions_with_scores
    
    This function does all the computational work that was previously done in the GPU stage:
    - Token position computation
    - Entity-to-token overlap detection
    - BIO tag assignment
    
    Args:
        raw_entities: List of entity dictionaries from Flair
        tokens: List of token strings
        source_text: Original source text
        entity_mapping: Mapping from Flair types to target types
    
    Returns:
        Tuple of (bio_labels, bio_scores)
    """
    bio_labels: List[str] = ["0"] * len(tokens)
    bio_scores: List[float] = [0.1] * len(tokens)  # Default low score for non-entities
    
    if not raw_entities:
        return bio_labels, bio_scores
    
    # Pre-compute token positions (heavy CPU work)
    token_positions = extract_token_positions(tokens, source_text)
    
    # Process each entity (heavy CPU work with nested loops)
    for entity_data in raw_entities:
        flair_type = entity_data['tag']
        target_type = entity_mapping.get(flair_type, None)
        
        if target_type is None:
            continue  # Skip unmapped entities
            
        # Get entity confidence, ensure it's in valid range
        entity_confidence = max(0.1, min(0.9, entity_data['score']))
        entity_start = entity_data['start_position']
        entity_end = entity_data['end_position']

        # Find overlapping tokens (heavy CPU work - nested loops)
        affected_tokens: List[int] = []
        for token_idx, (token_start_pos, token_end_pos) in enumerate(token_positions):
            # Check if this token overlaps with the entity
            if token_start_pos < entity_end and token_end_pos > entity_start:
                affected_tokens.append(token_idx)

        # Set BIO tags for affected tokens
        for i, token_idx in enumerate(affected_tokens):
            if 0 <= token_idx < len(bio_labels):
                if i == 0:
                    bio_labels[token_idx] = f"B-{target_type}"
                else:
                    bio_labels[token_idx] = f"I-{target_type}"
                bio_scores[token_idx] = entity_confidence
    
    return bio_labels, bio_scores


def create_entity_mapping(target_entity_types: List[str], dataset_name: Optional[str] = None) -> Dict[str, str]:
    """
    Create mapping from Flair entity types to target entity types using entity_label_mapping.json
    Only maps to labels that exist in the dataset, otherwise maps to '0'.
    
    This function loads the same mapping file used by the LSTM feature extractor to ensure
    consistency between ground truth processing and prediction processing.
    
    Args:
        target_entity_types: List of target entity types from config (for compatibility)
        dataset_name: Name of the dataset to get dataset-specific allowed labels
        
    Returns:
        Dictionary mapping Flair output types to final target types (only valid for dataset)
    """
    # Load the same mapping file used by LSTM feature extractor
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    mapping_file = os.path.join(project_root, 'data', 'entity_label_mapping.json')
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            base_mapping = mapping_data['mapping']
            
        # Get dataset-specific allowed labels
        if dataset_name and dataset_name in mapping_data.get('dataset_compatibility', {}):
            allowed_labels = set(mapping_data['dataset_compatibility'][dataset_name])
        else:
            # Fallback to target_entity_types from config
            allowed_labels = set(target_entity_types)
            
        # Create Flair-specific mappings for common Flair output types
        flair_to_target = {}
        
        # Map Flair entity types to our standardized types using the JSON mapping
        flair_mapping = {
            # Flair OntoNotes model commonly outputs these types
            'PERSON': base_mapping.get('PERSON_NAME', 'PERSON_NAME'),
            'PER': base_mapping.get('PERSON_NAME', 'PERSON_NAME'),
            'ORGANIZATION': base_mapping.get('ORGANIZATION', 'ORGANIZATION'),
            'ORG': base_mapping.get('ORGANIZATION', 'ORGANIZATION'),
            'GPE': base_mapping.get('GPE', 'GPE'),
            'LOC': base_mapping.get('LOCATION', 'LOCATION'),
            'FAC': base_mapping.get('LOCATION', 'LOCATION'),  # Facilities = locations/addresses
            'DATE': base_mapping.get('DATE', 'DATE'),
            'TIME': base_mapping.get('DATE', 'DATE'),  # Time entities often treated as dates
            'NORP': base_mapping.get('NORP', 'NORP'),  # Nationalities, etc.
            'MONEY': base_mapping.get('MISC', '0'),
            'PERCENT': base_mapping.get('MISC', '0'),
            'CARDINAL': base_mapping.get('MISC', '0'),  # Numbers (phone, SSN, etc.)
            'ORDINAL': base_mapping.get('MISC', '0'),
            'QUANTITY': base_mapping.get('MISC', '0'),
            'WORK_OF_ART': base_mapping.get('MISC', '0'),
            'EVENT': base_mapping.get('MISC', '0'),
            'PRODUCT': base_mapping.get('MISC', '0'),
            'LAW': base_mapping.get('MISC', '0'),
            'LANGUAGE': base_mapping.get('MISC', '0'),
        }
        
        # Apply fallback mapping for unmapped entities
        fallback = mapping_data.get('unmapped_fallback', '0')
        
        # Build final mapping, only using labels valid for this dataset
        for flair_type, target_type in flair_mapping.items():
            # Check if target type is allowed for this dataset
            if target_type in allowed_labels:
                flair_to_target[flair_type] = target_type
            else:
                # Map to fallback if not valid for this dataset
                flair_to_target[flair_type] = fallback
        
        return flair_to_target
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        # Fallback to basic mapping if JSON file is not available
        logger.warning("Could not load entity mapping file: %s", e)
        return {
            'PERSON': 'PERSON_NAME',
            'PER': 'PERSON_NAME', 
            'ORG': 'ORGANIZATION',
            'GPE': 'GPE',
            'LOC': 'LOCATION',
            'FAC': 'LOCATION',
            'DATE': 'DATE',
            'TIME': 'DATE',
            'NORP': 'NORP',
            'MONEY': '0',
            'PERCENT': '0',
            'CARDINAL': '0',
            'ORDINAL': '0',
            'QUANTITY': '0',
            'WORK_OF_ART': '0',
            'EVENT': '0',
            'PRODUCT': '0',
            'LAW': '0',
            'LANGUAGE': '0'
        }


def convert_bio_to_spans_worker(
    data_item: Dict[str, Any]
) -> Tuple[Optional[List[Span]], Optional[Union[List[Span], str]]]:
    """
    Simplified worker function that only converts BIO tags to spans
    
    This function expects the input data to already contain properly aligned BIO tags
    for both predictions and ground truth. It only performs the final conversion to spans.
    
    Args:
        data_item: Dictionary containing:
            - predicted_bio_tags: List of BIO labels for predictions
            - true_bio_tags: List of BIO labels for ground truth  
            - pred_scores: List of prediction confidence scores
            - tokens: List of token strings
            - source_text: Original text
            - sample_idx: Sample index for error reporting
    
    Returns:
        Tuple of (predicted_spans, true_spans) or (None, error_message)
    """
    try:
        predicted_bio_tags = data_item['predicted_bio_tags']
        true_bio_tags = data_item['true_bio_tags']
        pred_scores = data_item['pred_scores']
        tokens = data_item['tokens']
        source_text = data_item['source_text']
        
        # Convert predicted BIO tags to spans
        pred_spans = bio_to_spans(predicted_bio_tags, pred_scores, tokens, source_text)
        
        # Convert ground truth BIO tags to spans (use uniform score for ground truth)
        true_scores = [1.0] * len(true_bio_tags)
        true_spans = bio_to_spans(true_bio_tags, true_scores, tokens, source_text)
        
        return pred_spans, true_spans
        
    except (KeyError, TypeError, ValueError) as e:
        return None, f"Error processing sample {data_item.get('sample_idx', 'unknown')}: {e}"


def spans_to_evaluator_format(
    pred_spans: List[Span],
    true_spans: List[Span],
    num_tokens: int,
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Convert spans to the format expected by the evaluator
    
    The evaluator expects token-level predictions:
    - Predictions: List of (label, score) tuples for each token position
    - Ground truth: List of label strings for each token position
    
    Args:
        pred_spans: List of predicted span dictionaries
        true_spans: List of ground truth span dictionaries
        num_tokens: Number of tokens in the sequence
        
    Returns:
        Tuple of (pred_tokens, true_tokens) in evaluator format
    """
    # Initialize token-level predictions with default non-entity
    pred_tokens: List[Tuple[str, float]] = [("0", 0.1)] * num_tokens
    true_tokens: List[str] = ["0"] * num_tokens
    
    # Map predicted spans to token positions
    # Note: This is a simplified mapping - in reality we'd need token positions
    # But since we have the spans, we can make reasonable estimates
    for span in pred_spans:
        label = span['label']
        score = span.get('score', 1.0)
        
        # Estimate token positions from character positions
        # This is approximate but should work for evaluation
        start_char = span.get('start', 0)
        end_char = span.get('end', 0)
        
        # Convert character positions to approximate token positions
        # Assuming average ~5 characters per token
        start_token = max(0, min(start_char // 5, num_tokens - 1))
        end_token = max(start_token + 1, min(end_char // 5, num_tokens))
        
        # Set predictions for tokens in span
        for token_idx in range(start_token, min(end_token, num_tokens)):
            pred_tokens[token_idx] = (label, score)
    
    # Map ground truth spans to token positions
    for span in true_spans:
        label = span['label']
        
        # Same character-to-token mapping for ground truth
        start_char = span.get('start', 0)
        end_char = span.get('end', 0)
        
        start_token = max(0, min(start_char // 5, num_tokens - 1))
        end_token = max(start_token + 1, min(end_char // 5, num_tokens))
        
        # Set ground truth for tokens in span
        for token_idx in range(start_token, min(end_token, num_tokens)):
            true_tokens[token_idx] = label
    
    return pred_tokens, true_tokens


def extract_token_positions(tokens: List[str], source_text: str) -> List[Tuple[int, int]]:
    """
    Extract character positions for tokens in source text
    
    Args:
        tokens: List of token strings
        source_text: Original source text
        
    Returns:
        List of (start, end) character positions for each token
    """
    # Extract raw token text
    raw_tokens = []
    for token in tokens:
        if isinstance(token, str):
            if token.startswith("Token[") and ']: "' in token:
                start = token.find(']: "') + 4
                end = token.rfind('"')
                raw_tokens.append(token[start:end])
            else:
                raw_tokens.append(token)
        else:
            raw_tokens.append(str(token))
    
    # Create token position mapping
    token_positions = []
    current_char_pos = 0
    
    for token_text in raw_tokens:
        # Skip whitespace
        while (
            current_char_pos < len(source_text)
            and source_text[current_char_pos].isspace()
        ):
            current_char_pos += 1
        
        # Find token position
        token_start = source_text.find(token_text, current_char_pos)
        if token_start != -1:
            token_end = token_start + len(token_text)
            token_positions.append((token_start, token_end))
            current_char_pos = token_end
        else:
            # Fallback: approximate position
            token_positions.append(
                (current_char_pos, current_char_pos + len(token_text))
            )
            current_char_pos += len(token_text)
    
    return token_positions


def align_spans_to_bio_tags(spans: List[Any], tokens: List[str], source_text: str) -> List[str]:
    """
    Convert span labels to BIO tags aligned with model tokenization
    
    This is the crucial alignment step that ensures ground truth spans are properly
    aligned with the model's specific tokenization.
    
    Args:
        spans: List of span tuples/lists with [start, end, label] or dicts with 'start', 'end', 'label'
        tokens: List of model tokens  
        source_text: Original source text
        
    Returns:
        List of BIO labels aligned with the model tokens
    """
    # Get token positions
    token_positions = extract_token_positions(tokens, source_text)
    
    # Initialize BIO labels
    bio_labels: List[str] = ["O"] * len(tokens)
    
    # Convert each span to BIO tags
    for span in spans:
        # Handle both dictionary and list/tuple formats
        if isinstance(span, dict):
            span_start = span['start']
            span_end = span['end']
            entity_type = span['label']
        elif isinstance(span, (list, tuple)) and len(span) >= 3:
            span_start = span[0]
            span_end = span[1]
            entity_type = span[2]
        else:
            continue  # Skip malformed spans
        
        first_token = True
        for token_idx, (token_start, token_end) in enumerate(token_positions):
            # Check if token overlaps with span
            if token_start < span_end and token_end > span_start:
                if first_token:
                    bio_labels[token_idx] = f"B-{entity_type}"
                    first_token = False
                else:
                    bio_labels[token_idx] = f"I-{entity_type}"
    
    return bio_labels
