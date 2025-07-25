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
            from dataset_loader.ai4privacy_pii_data_loader import Ai4privacyPiiDataLoader

            return Ai4privacyPiiDataLoader(self.config, logger)
        
        if self.dataset_name == 'Conll2003':
            from dataset_loader.conll2003_data_loader import Conll2003DataLoader
            
            return Conll2003DataLoader(self.config, logger)
        
        if self.dataset_name == 'AFD':
            from dataset_loader.afd_data_loader import AFDDataLoader
            
            return AFDDataLoader(self.config, logger)
        
        if self.dataset_name == 'CDR':
            from dataset_loader.cdr_data_loader import CDRDataLoader
            
            return CDRDataLoader(self.config, logger)
        
        if self.dataset_name == 'LeNERbr':
            from dataset_loader.lenerbr_data_loader import LeNERbrDataLoader
            
            return LeNERbrDataLoader(self.config, logger)
        
        if self.dataset_name == 'Ontonotes5':
            from dataset_loader.ter_otonotes5_data_loader import Ontonotes5DataLoader

            return Ontonotes5DataLoader(self.config, logger)

        else:
            raise ValueError(
                f"Unsupported dataset or feature generator: {self.dataset_name}, {self.feature_generator_name}")