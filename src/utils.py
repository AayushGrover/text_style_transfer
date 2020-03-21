import torch
from transformers import BertModel, BertTokenizer, pipeline

import config

class BertUtil():
    def __init__(self, pretrained_weights=config.pretrained_weights, max_length=config.max_length):
        self.pretrained_weights = pretrained_weights
        self.max_length = max_length
        self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)

    def _generate_input_ids(self, sentence):
        encoded_dict = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_length, pad_to_max_length=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        # shape(input_ids) = [1, max_length]
        return input_ids
    
    def _generate_cls_embedding(self, input_ids):
        with torch.no_grad():
            out = self.model(input_ids)[0]
            cls_embedding = out[:, 0, :]
            # shape(cls_embedding) = [1, hidden_dim]
            return cls_embedding.to(config.device)
    
    def generate_cls_embedding(self, sentence):
        input_ids = self._generate_input_ids(sentence)
        cls_embedding = self._generate_cls_embedding(input_ids)
        return cls_embedding


class SentimentAnalysisUtil():
    def __init__(self, SENTIMENTS=config.SENTIMENTS):
        # leverages a fine-tuned model on sst2, which is a GLUE task.
        self.nlp = pipeline('sentiment-analysis')
        self.SENTIMENTS = SENTIMENTS
    
    def _get_sentiment_label(self, sentence):
        result = self.nlp(sentence)
        sentiment_label = result[0]['label']
        return sentiment_label
    
    def _get_sentiment_vector(self, sentiment_label):
        vec = torch.zeros(len(self.SENTIMENTS), dtype=torch.int8, device=config.device)
        vec[self.SENTIMENTS[sentiment_label]] = 1
        vec = vec.unsqueeze(0)
        # shape(vec) = [1, len(self.SENTIMENTS)] = [1, 2]
        return vec
    
    def get_sentiment_vector(self, sentence):
        sentiment_label = self._get_sentiment_label(sentence)
        vec = self._get_sentiment_vector(sentiment_label)
        return vec
    
    def get_sentiment_vector_from_label(self, sentiment_label):
        return self._get_sentiment_vector(sentiment_label)

if __name__ == '__main__':
    sentiment_analysis_util = SentimentAnalysisUtil()
    sentence = 'Sad'
    vec = sentiment_analysis_util.get_sentiment_vector(sentence)
    print(vec)