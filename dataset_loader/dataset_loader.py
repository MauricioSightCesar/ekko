import pandas as pd
import ast
from collections import Counter
from abc import ABC, abstractmethod

class DatasetLoader(ABC):
    def __init__(self, config, logger):
        self.logger = logger
        self.phase = config.get('phase')
        self.sample_size = config.get('dataset', {}).get('sample_size', None)

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        raise NotImplementedError("This method should be implemented by subclasses.")
