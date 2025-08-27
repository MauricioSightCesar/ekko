import ast
import pandas as pd

from datasets import load_dataset

from dataset_loader.dataset_loader import DatasetLoader


class Ai4privacyPiiDataLoader(DatasetLoader):
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

        return df
    

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
            'CITY': 'ADDRESS',
            'STATE': 'ADDRESS',
            'COUNTRY': 'ADDRESS',
            'POSTCODE': 'ADDRESS',
            'STREET': 'ADDRESS',
            'BUILDING': 'ADDRESS',
            'GEOCOORD': 'ADDRESS',
            'SECADDRESS': 'ADDRESS',

            # Credential Data
            'PASS': 'PASSWORD',

            # Email Address
            'EMAIL': 'EMAIL',

            # Sex Life or Sexual Orientation
            'SEX': 'SEX'
        }
