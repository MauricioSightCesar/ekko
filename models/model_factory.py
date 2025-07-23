from models.GLiNER import GLiNERModel

class ModelFactory:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.model_config = config.get('model')
        self.model_name = self.model_config.get('name')

    def create_model(self):
        if self.model_name == "GLiNER":
            self.version = self.model_config.get('version')
            return GLiNERModel.from_pretrained(self.version)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
