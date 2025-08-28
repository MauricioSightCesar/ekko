import pandas as pd

from feature_generator.feature_generator import FeatureGenerator

class SpaCyFeatureGenerator(FeatureGenerator):
    def generate_features(self, dataset) -> pd.DataFrame:
        return dataset
