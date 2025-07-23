import pandas as pd
import numpy as np
import typing

from abc import ABC, abstractmethod

from dataset_loader.dataset_loader import DatasetLoader

class FeatureGenerator(ABC):
    def __init__(self, config: typing.Dict, logger, dataset_loader: DatasetLoader):
        self.logger = logger
        self.dataset_loader = dataset_loader


    @abstractmethod
    def generate_features(self, labels: pd.DataFrame) -> pd.Series:
        pass
    

    def run(self):
        self.logger.info("Starting feature pipeline...")

        # Step 1: Load dataset
        raw_data = self.dataset_loader.load_raw()

        self.logger.info(f"Data loaded successfully.")

        # Step 2: Generate features
        data = self.generate_features(raw_data)

        self.logger.info("Pipeline complete.")

        return data


    def load_processed(self) -> tuple[np.ndarray, pd.DataFrame]:
        return self.run()
