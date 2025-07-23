import pandas as pd

from abc import ABC, abstractmethod

class DatasetLoader(ABC):
    def __init__(self, config, logger):
        self.logger = logger

        self.phase = config.get('phase')

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        raise NotImplementedError("This method should be implemented by subclasses.")
