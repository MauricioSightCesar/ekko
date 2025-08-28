import ast
import pandas as pd

import xml.etree.ElementTree as ET

from dataset_loader.dataset_loader import DatasetLoader


class CDRDataLoader(DatasetLoader):
    def load_raw(self) -> pd.DataFrame:
        """
        Load the CDR.Corpus.v010516 dataset.
        Returns:
            pd.DataFrame: A DataFrame containing the dataset with 'source_text' and 'span_labels' columns.
        """

        # Log the loading process
        self.logger.info("Loading CDR.Corpus.v010516 dataset...")

        # Parse the XML file
        phase = 'Training' if self.phase == 'train' else 'Test'
        tree = ET.parse(f'./data/CDR_Data/CDR.Corpus.v010516/CDR_{phase}Set.BioC.xml')
        root = tree.getroot()

        output = []

        for doc in root.findall('document'):
            for passage in doc.findall('passage'):
                source_text = passage.findtext('text')
                if not source_text:
                    continue

                spans = []
                for annotation in passage.findall('annotation'):
                    label = annotation.find("infon[@key='type']").text
                    location = annotation.find('location')
                    start = int(location.attrib['offset'])
                    length = int(location.attrib['length'])
                    end = start + length

                    spans.append([start, end, label])

                output.append({
                    "source_text": source_text,
                    "span_labels": spans
                })

        df = pd.DataFrame(output)
        df['language'] = 'English'

        df['span_labels'] = df['span_labels'].apply(self.filter_labels)
        
        # Log the number of rows loaded
        self.logger.info(f"Loaded {len(df)} rows for phase '{self.phase}'")

        return df
    

    def filter_labels(self, span_labels):
        """
        Filter out unwanted labels from the span_labels.
        """
        use_labels = ['Disease']
        mapping = self.entities_mapping()

        return [[span[0], span[1], mapping[span[2]]] for span in span_labels if span[2] in use_labels]
    
    
    def entities_mapping(self) -> dict:
        return {
            'Disease': 'DISEASE',
        }
