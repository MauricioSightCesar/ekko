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
            from models.spaCy import SpaCy
            return SpaCy(self.config, self.logger)
        
        elif self.model_name == "LSTMCRF":
            from models.pretrained_lstm_crf_flair import PretrainedLSTMCRFFlairModel
            device = self.config.get('device', 'cpu')
            model_name = self.model_config.get('pretrained_model', "flair/ner-english-ontonotes")
            return PretrainedLSTMCRFFlairModel(device=device, model_name=model_name, logger=self.logger)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")
