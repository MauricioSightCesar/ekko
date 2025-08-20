class FeatureGeneratorFactory:
    def __init__(self, config):
        self.config = config

        self.feature_config = config.get('feature')
        self.feature_generator_name = self.feature_config.get('generator')

    def get_feature_generator(self, logger, dataset_loader):
        if self.feature_generator_name == 'GLiNER':
            from feature_generator.gliner_feature_extractor import GLiNERFeatureGenerator
            return GLiNERFeatureGenerator(self.config, logger, dataset_loader)
        
        if self.feature_generator_name == 'spaCy':
            from feature_generator.spacy_feature_extractor import SpaCyFeatureGenerator
            return SpaCyFeatureGenerator(self.config, logger, dataset_loader)

        else:
            raise ValueError(
                f"Unsupported feature generator: {self.feature_generator_name}")