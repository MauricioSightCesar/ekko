import ast
import pandas as pd
from collections import Counter
import numpy as np

from datasets import load_dataset

from dataset_loader.dataset_loader import DatasetLoader


class Ai4privacyPiiDataLoader(DatasetLoader):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.config = config  # Store full config for access to dataset.sample_size
    
    def load_raw(self) -> pd.DataFrame:
        """
        Load the ai4privacy/pii-masking-300k dataset.
        Returns:
            pd.DataFrame: A DataFrame containing the dataset with 'source_text' and 'span_labels' columns.
        """

        # Log the loading process
        self.logger.info("Loading ai4privacy/pii-masking-300k dataset...")

        # Load the dataset using the specified phase
        ds = load_dataset("ai4privacy/pii-masking-300k", split=self.phase)

        df = ds.to_pandas()

        df.drop(columns=["target_text", 'mbert_text_tokens', 'privacy_mask', 'mbert_bio_labels', 'id', 'set'], inplace=True)

        df['span_labels'] = df['span_labels'].apply(self.filter_labels)
        
        # Log the number of rows loaded
        self.logger.info(f"Loaded {len(df)} rows for phase '{self.phase}'")

        # Apply sampling and analyze label distributions
        df = self.apply_sampling_and_analyze(df)
        
        return df
    
    def apply_sampling_and_analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sampling based on config and analyze label distributions.
        """
        # Get sample_size from config
        sample_size = self.config.get('dataset', {}).get('sample_size', None)
        
        # Analyze full dataset first
        full_label_counts = self.count_labels(df)
        total_labels_full = sum(full_label_counts.values())
        
        # Apply sampling if specified
        if sample_size is not None and sample_size < len(df):
            self.logger.info(f"Sampling {sample_size} from {len(df)} total samples")
            
            # Random sampling
            df_sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # Analyze sampled dataset
            sample_label_counts = self.count_labels(df_sampled)
            total_labels_sample = sum(sample_label_counts.values())
            
            # Compare full vs sampled
            self.logger.info(f"\n=== SAMPLING COMPARISON ===")
            self.logger.info("Label preservation (sampled/full):")
            
            for label in sorted(set(full_label_counts.keys()) | set(sample_label_counts.keys())):
                full_count = full_label_counts.get(label, 0)
                sample_count = sample_label_counts.get(label, 0)
                
                if full_count > 0:
                    preservation_ratio = (sample_count / full_count) * 100
                    sample_percentage = (sample_count / total_labels_sample) * 100 if total_labels_sample > 0 else 0
                    full_percentage = (full_count / total_labels_full) * 100
                    
                    self.logger.info(f"  {label}: {sample_count} - {sample_percentage:.2f}% / {full_count} - {full_percentage:.2f}% (Preservation: {preservation_ratio:.2f}%)")
                else:
                    self.logger.info(f"  {label}: Not present in full dataset")
            
            return df_sampled
        else:
            if sample_size is None:
                self.logger.info("No sampling applied (sample_size is null)")
            else:
                self.logger.info(f"No sampling needed (sample_size {sample_size} >= dataset size {len(df)})")
            return df
    
    def count_labels(self, df: pd.DataFrame) -> dict:
        """
        Count occurrences of each label in the dataset.
        """
        label_counts = Counter()
        
        for _, row in df.iterrows():
            span_labels = row['span_labels']
            if isinstance(span_labels, list):
                for span in span_labels:
                    if len(span) >= 3:  # Ensure span has label at index 2
                        label = span[2]
                        label_counts[label] += 1
        
        return dict(label_counts)
    

    def filter_labels(self, span_labels):
        """
        Filter out unwanted labels from the span_labels.
        """

        exclude_labels = ['TIME', 'DATE', 'IP',  'TITLE']
        mapping = self.entities_mapping()

        return [[span[0], span[1], mapping[span[2]]] for span in ast.literal_eval(span_labels) if span[2] not in exclude_labels]
    
    
    def entities_mapping(self) -> dict:
        return {
            # Name
            'GIVENNAME1': 'PERSON_NAME',
            'GIVENNAME2': 'PERSON_NAME',
            'LASTNAME1': 'PERSON_NAME',
            'LASTNAME2': 'PERSON_NAME',
            'LASTNAME3': 'PERSON_NAME',
            'USERNAME': 'PERSON_NAME',

            # Identification Number
            'TEL': 'PHONE_NUMBER',
            'SOCIALNUMBER': 'SOCIAL_NUMBER',
            'DRIVERLICENSE': 'DRIVER_LICENSE',
            'IDCARD': 'ID_CARD',
            'PASSPORT': 'PASSPORT',
            'CARDISSUER': 'CARD_ISSUER',

            # Identification Date
            'BOD': 'BIRTHDATE',   # Birthdate

            # Location Data
            'CITY': 'CITY',
            'STATE': 'STATE',
            'COUNTRY': 'COUNTRY',
            'POSTCODE': 'POSTCODE',
            'STREET': 'STREET',
            'BUILDING': 'BUILDING_NUMBER',
            'GEOCOORD': 'GEO_COORD',
            'SECADDRESS': 'SEC_ADDRESS',

            # Credential Data
            'PASS': 'PASSWORD',

            # Email Address
            'EMAIL': 'EMAIL',

            # Sex Life or Sexual Orientation
            'SEX': 'SEX'
        }
