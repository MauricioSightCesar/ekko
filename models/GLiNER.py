import torch
import json

import torch

from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
import tqdm

from utils.experiment_io import get_run_dir

class GLiNERModel(GLiNER):
    @classmethod
    def from_pretrained(self, *args, config=None, logger=None):
        self.config = config
        self.logger = logger

        self.run_dir = get_run_dir(config.get('run_id'))
        self.entity_types = self.config.get('feature', {}).get('labels', [])

        self.batch_size = self.config.get('model', {}).get('batch_size', [])
        self.threshold = 0

        return super(GLiNERModel, self).from_pretrained(*args)

    def evaluate(
        self,
        test_data,
        flat_ner=True,
        multi_label=False,
    ):

        """
        Evaluate the model on a given test dataset.

        Args:
            test_data (List[Dict]): The test data containing text and entity annotations.
            flat_ner (bool): Whether to use flat NER. Defaults to False.
            multi_label (bool): Whether to use multi-label classification. Defaults to False.
            threshold (float): The threshold for predictions. Defaults to 0.5.
            batch_size (int): The batch size for evaluation. Defaults to 12.
            entity_types (Optional[List[str]]): List of entity types to consider. Defaults to None.

        Returns:
            tuple: A tuple containing the evaluation output and the F1 score.
        """
        self.eval()

        span_labels = test_data['span_labels'].tolist()
        source_text = test_data['source_text'].tolist()

        # Transform data into models input
        input_x, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_texts(source_text)

        # Get y_true -> labels per token
        y_true = self.__get_labels_for_token(span_labels, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx)

        collator = DataCollator(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
            entity_types=self.entity_types,
        )
        data_loader = torch.utils.data.DataLoader(
            input_x, batch_size=self.batch_size, shuffle=False, collate_fn=collator
        )

        y_pred = []

        # Iterate over data batches
        for batch_idx, batch in enumerate(data_loader):
            # Move the batch to the appropriate device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Perform predictions
            model_output = self.model(**batch)[0]

            if not isinstance(model_output, torch.Tensor):
                model_output = torch.from_numpy(model_output)

            decoded_outputs = self.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                model_output,
                flat_ner=flat_ner,
                threshold=self.threshold,
                multi_label=multi_label,
            )

            if batch_idx % 100 == 0 or batch_idx * len(batch) == len(data_loader.dataset) - 1:
                self.logger.info('Batch: [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(batch), len(data_loader.dataset), 
                    100. * batch_idx / len(data_loader)))
                
            with open(self.run_dir / "y_pred.json", "a") as f:
                for pred in decoded_outputs:
                    json.dump(pred, f)
                    f.write("\n")

            y_pred.extend(decoded_outputs)

        with open(self.run_dir / "y_true.json", "a") as f:
            json.dump(y_true, f)
            f.write("\n")

        return self.__convert_span_to_tokens(y_pred), y_true
    
    def __get_labels_for_token(self, span_labels, start_tokens_indexes, end_tokens_indexes):
        y_true = []
        for ex_idx in range(len(span_labels)):
            span_label = span_labels[ex_idx]
            start_tokens = start_tokens_indexes[ex_idx]
            end_tokens = end_tokens_indexes[ex_idx]
                
            example_label = []

            for start_token_idx, end_token_idx in zip(start_tokens, end_tokens):
                updated = False
                    
                for start_label_idx, end_label_idx, label in span_label:
                    if end_label_idx < start_token_idx:
                        continue

                    if ((start_label_idx < end_token_idx and start_label_idx >= start_token_idx) or 
                        (end_label_idx > start_token_idx and end_label_idx <= end_token_idx) or
                        (start_token_idx >= start_label_idx and end_token_idx <= end_label_idx)):
                        example_label.append(label)
                        updated = True
                        continue
                    
                if not updated:
                    example_label.append('0')

            y_true.append(example_label)
            
        return y_true
    
    def __get_span_token_labels(self, label_tokens):
        y_true = []

        for example in label_tokens:
            example_label = []

            start_index = 0
            for i in range(1, len(example)):
                label = example[i]

                if label == example[i-1]:
                    continue

                if example[i-1] != '0':
                    example_label.append((start_index, i-1, example[i-1]))
                
                start_index = i
            
            if example[-1] != '0':
                example_label.append((start_index, len(example) - 1, example[-1]))

            y_true.append(example_label)

        return y_true

    def __convert_span_to_tokens(self, y_pred):
        ds_label_per_token = []

        for sentence in y_pred:
            label_per_token = []

            for span in sentence:
                n_tokens = span[1] - span[0] + 1
                for _ in range(n_tokens):
                    label_per_token.append((span[2], span[3]))
                
            ds_label_per_token.append(label_per_token)
        
        return ds_label_per_token

    def __decode(self, tokens, id_to_classes, model_output, flat_ner=False, threshold=0.5, multi_label=False):
        probs = torch.sigmoid(model_output)
        spans = []
        for i, _ in enumerate(tokens):
            probs_i = probs[i]
            
            # Support for id_to_classes being a list of dictionaries
            id_to_class_i = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
            
            wh_i = [i.tolist() for i in torch.where(probs_i > threshold)]
            span_i = []
            for s, k, c in zip(*wh_i):
                if s + k < len(tokens[i]):
                    span_i.append((s, s + k, id_to_class_i[c + 1], probs_i[s, k, c].item()))

            # span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)
            spans.append(span_i)
        return spans