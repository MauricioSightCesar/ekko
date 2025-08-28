import ast
import pandas as pd

from dataset_loader.dataset_loader import DatasetLoader


class AFDDataLoader(DatasetLoader):
    def load_raw(self) -> pd.DataFrame:
        """
        Load the Synthetic Dataset for PII Detection and Anonymization in Financial Documents dataset.
        Returns:
            pd.DataFrame: A DataFrame containing the dataset with 'source_text' and 'span_labels' columns.
        """

        # Log the loading process
        self.logger.info("Loading AFD dataset...")

        # Load the dataset using the specified phase
        phase = 'Training' if self.phase == 'train' else 'Testing'
        df = pd.read_excel(f"data/AFD/{phase}_Set.xlsx")

        df.drop(columns=["Name", 'Credit Card', 'Email', 'URL', 'Phone', 'Address', 'Company', 'SSN'], inplace=True)

        df.rename(columns={'Text': 'source_text', 'True Predictions': 'span_labels'}, inplace=True)

        df['span_labels'] = df['span_labels'].apply(self.filter_labels)
        
        # Log the number of rows loaded
        self.logger.info(f"Loaded {len(df)} rows for phase '{self.phase}'")

        return df
    

    def filter_labels(self, span_labels):
        """
        Filter out unwanted labels from the span_labels.
        """

        exclude_labels = ['url', 'company', 'phone', 'ssn', 'credit_card', 'email']
        mapping = self.entities_mapping()

        return [[span[0], span[1], mapping[span[2]]] for span in ast.literal_eval(span_labels) if span[2] not in exclude_labels]
    
    
    def entities_mapping(self) -> dict:
        return {
            # Name
            'name': 'PERSON_NAME',

            # Identification Number
            'phone': 'PHONE_NUMBER',
            'ssn': 'SOCIAL_NUMBER',
            'credit_card': 'CARD_ISSUER',

            # Location Data
            'address': 'ADDRESS',

            # Email Address
            'email': 'EMAIL'
        }
