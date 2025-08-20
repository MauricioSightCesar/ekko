import spacy

class SpaCy:
    def __init__(self, config):
        self.config = config
        
        self.model_config = config.get('model')
        self.version = self.model_config.get('version')
        
        self.nlp = spacy.load(self.version)
        self.entity_types = self.config.get('feature', {}).get('labels', [])

    def compile(self, device):
        return self

    def evaluate(
        self,
        test_data,
    ):
        span_labels = test_data['span_labels'].tolist()
        source_text = test_data['source_text'].tolist()
        
        y_pred = []
        y_true = []

        for i, sentence in enumerate(source_text):
            span_label = span_labels[i]
            doc = self.nlp(sentence)

            tokens = []
            labels = []
            for d in doc:
                # Text, BIO label, Entity type, char start idx, char end idx
                label = self.map_labels(d.ent_type_)

                tokens.append((label, 0 if label == '0' else 1))

                start_token_idx = d.idx
                end_token_idx = d.idx+len(d)

                for start_label_idx, end_label_idx, label in span_label:
                    if end_label_idx < start_token_idx:
                        continue

                    if ((start_label_idx < end_token_idx and start_label_idx >= start_token_idx) or 
                        (end_label_idx > start_token_idx and end_label_idx <= end_token_idx) or
                        (start_token_idx >= start_label_idx and end_token_idx <= end_label_idx)):
                        labels.append(label)
                        break
                
                if len(tokens) > len(labels):
                    labels.append('0')
                    
            y_pred.append(tokens)
            y_true.append(labels)

        
        return y_pred, y_true
    
    def map_labels(self, label):
        return {
            'FAC': 'ADDRESS',
            'LOC': 'ADDRESS',
            'GPE': 'GEOPOLITICAL_ENTITIES',
            'NORP': 'NATIONAL_ORIGIN RACE PROTECTED_GROUP',
            'PERSON': 'PERSON_NAME',
            'CARDINAL': '0', 
            'DATE': '0', 
            'EVENT': '0', 
            'LANGUAGE': '0', 
            'LAW': '0', 
            'MONEY': '0',  
            'ORDINAL': '0', 
            'ORG': '0', 
            'PERCENT': '0', 
            'PRODUCT': '0', 
            'QUANTITY': '0', 
            'TIME': '0', 
            'WORK_OF_ART': '0',
            '': '0'
        }[label]

