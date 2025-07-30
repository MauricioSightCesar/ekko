
from feature_generator.gliner_feature_extractor import GLiNERFeatureGenerator
from feature_generator.lstmcrf_feature_extractor import LSTMCRFFeatureGenerator

class FeatureGeneratorFactory:
    def __init__(self, config):
        self.config = config

        self.feature_config = config.get('feature')
        self.feature_generator_name = self.feature_config.get('generator')

    def get_feature_generator(self, logger, dataset_loader):
        if self.feature_generator_name == 'GLiNER':
            return GLiNERFeatureGenerator(self.config, logger, dataset_loader)
        elif self.feature_generator_name == 'LSTMCRF':
            return LSTMCRFFeatureGenerator(self.config, logger, dataset_loader)
        else:
            raise ValueError(
                f"Unsupported feature generator: {self.feature_generator_name}")
