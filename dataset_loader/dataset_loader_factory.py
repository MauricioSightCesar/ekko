from dataset_loader.afd_data_loader import AFDDataLoader
from dataset_loader.cdr_data_loader import CDRDataLoader
from dataset_loader.lenerbr_data_loader import LeNERbrDataLoader
from dataset_loader.conll2003_data_loader import Conll2003DataLoader
from dataset_loader.ter_otonotes5_data_loader import Ontonotes5DataLoader
from dataset_loader.ai4privacy_pii_data_loader import Ai4privacyPiiDataLoader

class DatasetLoaderFactory:
    """
    Base class for dataset factories.
    This class provides a template for loading and processing datasets.
    """

    def __init__(self, config):
        self.config = config

        self.dataset_config = config.get('dataset')
        self.dataset_name = self.dataset_config.get('name')

    def get_dataset_loader(self, logger):
        if self.dataset_name == 'Ai4privacyPii':
            return Ai4privacyPiiDataLoader(self.config, logger)
        
        if self.dataset_name == 'Conll2003':
            return Conll2003DataLoader(self.config, logger)
        
        if self.dataset_name == 'AFD':
            return AFDDataLoader(self.config, logger)
        
        if self.dataset_name == 'CDR':
            return CDRDataLoader(self.config, logger)
        
        if self.dataset_name == 'LeNERbr':
            return LeNERbrDataLoader(self.config, logger)
        
        if self.dataset_name == 'Ontonotes5':
            return Ontonotes5DataLoader(self.config, logger)

        else:
            raise ValueError(
                f"Unsupported dataset or feature generator: {self.dataset_name}, {self.feature_generator_name}")