import ast
import pandas as pd

from datasets import load_dataset

from dataset_loader.dataset_loader import DatasetLoader


class Conll2003DataLoader(DatasetLoader):
    def load_raw(self) -> pd.DataFrame:
        """
        Load the Conll2003 dataset.
        Returns:
            pd.DataFrame: A DataFrame containing the dataset with 'source_text' and 'span_labels' columns.
        """

        # Log the loading process
        self.logger.info("Loading Conll2003 dataset...")

        # Load the dataset using the specified phase
        ds = load_dataset("eriktks/conll2003", split=self.phase, trust_remote_code=True)

        converted_data = []
        for example in ds:
            converted = self.bio_to_spans(example['tokens'], example['ner_tags'])
            converted_data.append(converted)

        df = pd.DataFrame(converted_data)
        df['language'] = 'English'

        df['span_labels'] = df['span_labels'].apply(self.filter_labels)
        
        # Log the number of rows loaded
        self.logger.info(f"Loaded {len(df)} rows for phase '{self.phase}'")

        return df
    

    def filter_labels(self, span_labels):
        """
        Filter out unwanted labels from the span_labels.
        """

        use_labels = ['person', 'location']
        mapping = self.entities_mapping()

        return [[span[0], span[1], mapping[span[2]]] for span in span_labels if span[2] in use_labels]
    
    
    def entities_mapping(self) -> dict:
        return {
            # Name
            'person': 'PERSON_NAME',

            # Location Data
            'location': 'LOCATION',

            'organization': 'ORGANIZATION',
            'others': 'OTHERS'
        }

    def bio_to_spans(self, tokens, tags):
        label2id = {
            'O': 0, 'B-person': 1, 'I-person': 2, 'B-organization': 3, 'I-organization': 4,
            'B-location': 5, 'I-location': 6, 'B-others': 7, 'I-others': 8
        }

        id2label = {v: k for k, v in label2id.items()}

        source_text = ""
        span_labels = []

        idx_map = []  # Maps token index to character start in source_text
        current_pos = 0

        # Build the source_text and index map
        for token in tokens:
            if source_text:
                source_text += " "
                current_pos += 1
            idx_map.append(current_pos)
            source_text += token
            current_pos += len(token)

        current_label = None
        start_token = None

        for i, tag in enumerate(tags):
            label = id2label[tag]

            if label == "O" or label == 0:
                if current_label:
                    span_labels.append([idx_map[start_token], idx_map[i - 1] + len(tokens[i - 1]), current_label])
                    current_label = None
                    start_token = None
            elif label.startswith("B-"):
                if current_label:
                    span_labels.append([idx_map[start_token], idx_map[i - 1] + len(tokens[i - 1]), current_label])
                current_label = label[2:]
                start_token = i
            elif label.startswith("I-") and current_label == label[2:]:
                continue
            else:
                # In case of misaligned I- without a B-, treat as B-
                current_label = label[2:]
                start_token = i

        # Final entity
        if current_label:
            span_labels.append([idx_map[start_token], idx_map[len(tokens) - 1] + len(tokens[-1]), current_label])

        return {
            "source_text": source_text,
            "span_labels": span_labels
        }
