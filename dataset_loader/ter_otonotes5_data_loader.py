import ast
import pandas as pd

from datasets import load_dataset

from dataset_loader.dataset_loader import DatasetLoader


class Ontonotes5DataLoader(DatasetLoader):
    def load_raw(self) -> pd.DataFrame:
        """
        Load the tner/ontonotes5 dataset.
        Returns:
            pd.DataFrame: A DataFrame containing the dataset with 'source_text' and 'span_labels' columns.
        """

        # Log the loading process
        self.logger.info("Loading tner/ontonotes5 dataset...")

        # Load the dataset using the specified phase
        ds = load_dataset("tner/ontonotes5", split=self.phase)

        converted_data = []
        for example in ds:
            converted = self.bio_to_spans(example['tokens'], example['tags'])
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

        use_labels = ['PERSON', 'NORP', 'GPE']
        mapping = self.entities_mapping()

        return [[span[0], span[1], mapping[span[2]]] for span in ast.literal_eval(span_labels) if span[2] not in use_labels]
    
    
    def entities_mapping(self) -> dict:
        return {
            # Name
            'PERSON': 'PERSON_NAME',

            # Racial, Ethnic Origin, Religious or Philosophical Beliefs
            'NORP': 'NORP',

            # Location Data
            'GPE': 'GPE',
        }

    def bio_to_spans(tokens, tags):
        label2id = {
            "O": 0, "B-CARDINAL": 1, "B-DATE": 2, "I-DATE": 3, "B-PERSON": 4, "I-PERSON": 5,
            "B-NORP": 6, "B-GPE": 7, "I-GPE": 8, "B-LAW": 9, "I-LAW": 10, "B-ORG": 11, "I-ORG": 12,
            "B-PERCENT": 13, "I-PERCENT": 14, "B-ORDINAL": 15, "B-MONEY": 16, "I-MONEY": 17,
            "B-WORK_OF_ART": 18, "I-WORK_OF_ART": 19, "B-FAC": 20, "B-TIME": 21, "I-CARDINAL": 22,
            "B-LOC": 23, "B-QUANTITY": 24, "I-QUANTITY": 25, "I-NORP": 26, "I-LOC": 27,
            "B-PRODUCT": 28, "I-TIME": 29, "B-EVENT": 30, "I-EVENT": 31, "I-FAC": 32,
            "B-LANGUAGE": 33, "I-PRODUCT": 34, "I-ORDINAL": 35, "I-LANGUAGE": 36
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
