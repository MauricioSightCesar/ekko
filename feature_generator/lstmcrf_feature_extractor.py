"""Feature generation for LSTM-CRF models.

This generator maps dataset span labels using a JSON mapping file so the
downstream model/evaluator operates on consistent label sets. No heavy
feature engineering is performed; we only normalize labels.
"""
from __future__ import annotations

import json
import os
import ast
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from feature_generator.feature_generator import FeatureGenerator

class LSTMCRFFeatureGenerator(FeatureGenerator):
    """Maps entity labels to a normalized set used by the LSTM-CRF pipeline."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, dataset_loader: Any) -> None:
        """Initialize the feature generator and load the entity mapping.

        Args:
            config: Configuration dictionary.
            logger: Logger instance for reporting.
            dataset_loader: Dataset loader instance (not used directly here).
        """
        super().__init__(config, logger, dataset_loader)
        self.mapping: Dict[str, str] = self._load_entity_mapping()
        
    def _load_entity_mapping(self) -> Dict[str, str]:
        """Load entity label mapping from JSON file.

        Returns:
            A dictionary mapping raw labels to normalized labels.
        """
        # Determine mapping file path relative to repo root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        mapping_file = os.path.join(project_root, 'data', 'LSTM_entity_label_mapping.json')

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                mapping = mapping_data['mapping']
                # basic validation
                if not isinstance(mapping, dict):
                    raise ValueError("'mapping' must be a dict in LSTM_entity_label_mapping.json")
                return {str(k): str(v) for k, v in mapping.items()}
        except FileNotFoundError:
            self.logger.warning("Entity mapping file not found: %s", mapping_file)
            return {}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning("Failed to load entity mapping: %s", e)
            return {}
    
    def generate_features(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Map span labels using the loaded mapping and return the updated DataFrame.

        The LSTM-CRF pipeline expects normalized labels. If the mapping is empty,
        the input is returned unchanged.

        Args:
            labels: DataFrame with a 'span_labels' column containing spans as
                list representations (or stringified lists).

        Returns:
            A DataFrame where 'span_labels' has labels mapped to the normalized set.
        """
        if not isinstance(labels, pd.DataFrame) or 'span_labels' not in labels.columns:
            self.logger.warning("generate_features expects a DataFrame with 'span_labels' column; returning input.")
            return labels

        if not self.mapping:
            # Keep logs minimal; a single warning is enough.
            self.logger.warning("No entity mapping loaded; labels remain unchanged.")
            return labels

        def _map_spans_cell(spans_cell: Any) -> List[List[Any]]:
            # Parse stringified spans if necessary
            spans_val: Any = spans_cell
            if isinstance(spans_val, str):
                try:
                    spans_val = ast.literal_eval(spans_val)
                except (ValueError, SyntaxError):
                    return []

            if not isinstance(spans_val, list):
                return []

            mapped: List[List[Any]] = []
            for item in spans_val:
                # Support list [start, end, label]
                if isinstance(item, list):
                    if len(item) >= 3:
                        start, end, label = item[0], item[1], item[2]
                        mapped_label = self.mapping.get(str(label), '0')
                        mapped.append([start, end, mapped_label])
                    elif len(item) == 2:
                        # Missing label => treat as no-entity
                        mapped.append([item[0], item[1], '0'])
                # Support dict {start, end, label}
                elif isinstance(item, dict):
                    try:
                        start = item.get('start')
                        end = item.get('end')
                        label = item.get('label', '0')
                        mapped_label = self.mapping.get(str(label), '0')
                        mapped.append([start, end, mapped_label])
                    except Exception:
                        # Be permissive; skip malformed entries quietly
                        continue
            return mapped

        labels = labels.copy()
        labels['span_labels'] = labels['span_labels'].apply(_map_spans_cell)
        return labels
