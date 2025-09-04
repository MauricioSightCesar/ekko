"""Feature generation for DeBERTa-based NER.

This generator maps dataset span labels to the subset the DeBERTa model can
predict using a JSON mapping file and the dataset_compatibility list.
Any label not supported for the current dataset (per mapping file) is mapped
to '0' so we only evaluate predictable labels.
"""
from __future__ import annotations

import ast
import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd

from feature_generator.feature_generator import FeatureGenerator


class DebertaFeatureGenerator(FeatureGenerator):
    """Maps entity labels to the set supported by the DeBERTa model."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, dataset_loader: Any) -> None:
        super().__init__(config, logger, dataset_loader)
        self.dataset_name: str = str(config.get("dataset", {}).get("name", ""))
        self.mapping, self.allowed_labels, self.fallback = self._load_entity_mapping()

    def _load_entity_mapping(self) -> tuple[Dict[str, str], List[str], str]:
        """Load DeBERTa entity label mapping and dataset compatibility.

        Returns:
            mapping: raw_label -> mapped_label
            allowed_labels: labels allowed for this dataset (predictable by model)
            fallback: fallback label for unmapped or unsupported labels
        """
        # Determine mapping file path relative to repo root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        mapping_file = os.path.join(project_root, 'data', 'DEBERTA_entity_label_mapping.json')

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            mapping = mapping_data.get('mapping', {})
            if not isinstance(mapping, dict):
                raise ValueError("'mapping' must be a dict in DEBERTA mapping JSON")
            fallback = str(mapping_data.get('unmapped_fallback', '0'))

            ds_comp = mapping_data.get('dataset_compatibility', {})
            allowed = ds_comp.get(self.dataset_name, []) if isinstance(ds_comp, dict) else []
            # Normalize to strings
            mapping = {str(k): str(v) for k, v in mapping.items()}
            allowed = [str(x) for x in allowed]
            return mapping, allowed, fallback
        except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.warning("Could not load DeBERTa mapping file: %s", e)
            return {}, [], '0'

    def generate_features(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Map span labels using the DeBERTa mapping and dataset compatibility.

        - First, map dataset/raw labels to DeBERTa-equivalent labels using `mapping`.
        - Then, if a mapped label is not in `allowed_labels` for this dataset,
          replace it with the fallback ('0').

        Args:
            labels: DataFrame with a 'span_labels' column containing spans as
                list representations (or stringified lists).

        Returns:
            A DataFrame where 'span_labels' has labels normalized and filtered.
        """
        if not isinstance(labels, pd.DataFrame) or 'span_labels' not in labels.columns:
            self.logger.warning("generate_features expects a DataFrame with 'span_labels' column; returning input.")
            return labels

        if not self.mapping:
            self.logger.warning("No DeBERTa entity mapping loaded; labels remain unchanged.")
            return labels

        allowed_set = set(self.allowed_labels) if self.allowed_labels else None
        fallback = self.fallback or '0'

        def _map_spans_cell(spans_cell: Any) -> List[List[Any]]:
            # Parse string to list if needed
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
                if isinstance(item, list):
                    if len(item) >= 3:
                        start, end, label = item[0], item[1], str(item[2])
                        mapped_label = self.mapping.get(label, fallback)
                        # Enforce dataset compatibility: only keep labels model can predict
                        if allowed_set is not None and mapped_label not in allowed_set:
                            mapped_label = fallback
                        mapped.append([start, end, mapped_label])
                    elif len(item) == 2:
                        mapped.append([item[0], item[1], fallback])
                elif isinstance(item, dict):
                    try:
                        start = item.get('start')
                        end = item.get('end')
                        label = str(item.get('label', fallback))
                        mapped_label = self.mapping.get(label, fallback)
                        if allowed_set is not None and mapped_label not in allowed_set:
                            mapped_label = fallback
                        mapped.append([start, end, mapped_label])
                    except Exception:
                        continue
            return mapped

        labels = labels.copy()
        labels['span_labels'] = labels['span_labels'].apply(_map_spans_cell)
        return labels
