class ModelFactory:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.model_config = config.get('model')
        self.model_name = self.model_config.get('name')

    def create_model(self):
        if self.model_name == "GLiNER":
            from models.GLiNER import GLiNERModel
            
            self.version = self.model_config.get('version')
            return GLiNERModel.from_pretrained(self.version, config=self.config, logger=self.logger)
        
        if self.model_name == "spaCy":
            from models import spaCy
            return spaCy(self.config)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
