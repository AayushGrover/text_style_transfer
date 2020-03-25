import torch
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, pipeline

import config

class BertUtil():
    def __init__(self, pretrained_weights=config.bert_pretrained_weights, max_length=config.max_length):
        self.pretrained_weights = pretrained_weights
        self.max_length = max_length
        self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.model.to(config.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)

    def _generate_input_ids(self, sentence):
        encoded_dict = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_length, pad_to_max_length=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids'].to(config.device)
        # shape(input_ids) = [1, max_length]
        return input_ids
    
    def _generate_cls_embedding(self, input_ids):
        with torch.no_grad():
            out = self.model(input_ids)[0]
            cls_embedding = out[:, 0, :].squeeze(0).to(config.device)
            # shape(cls_embedding) = [1, hidden_dim]
            return cls_embedding
    
    def _generate_word_embeddings(self, input_ids):
        with torch.no_grad():
            out = self.model(input_ids)[0].squeeze(0).to(config.device)
            return out
    
    def generate_cls_embedding(self, sentence):
        input_ids = self._generate_input_ids(sentence)
        cls_embedding = self._generate_cls_embedding(input_ids)
        return cls_embedding
    
    def generate_word_embeddings(self, sentence):
        input_ids = self._generate_input_ids(sentence)
        word_embeddings = self._generate_word_embeddings(input_ids)
        return word_embeddings


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
        vec = torch.zeros(len(self.SENTIMENTS), dtype=torch.float, device=config.device)
        vec[self.SENTIMENTS[sentiment_label]] = 1
        # shape(vec) = [len(self.SENTIMENTS)] = [2]
        return vec
    
    def get_sentiment_vector(self, sentence):
        sentiment_label = self._get_sentiment_label(sentence)
        vec = self._get_sentiment_vector(sentiment_label)
        return vec
    
    def get_sentiment_vector_from_label(self, sentiment_label):
        return self._get_sentiment_vector(sentiment_label)


class GPT2Util():
    def __init__(self, pretrained_weights=config.gpt2_pretrained_weights, max_length=config.max_length):
        self.pretrained_weights = pretrained_weights
        self.max_length = max_length
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_weights)
        self.model.to(config.device)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_weights)
    
    def batch_generate_sentence(self, inputs_embeds):
        with torch.no_grad():
            predictions = self.model(inputs_embeds=inputs_embeds)[0]
            # shape(predictions) = [batch_size, max_length, gpt2_vocab_size]

        batch_predicted_indices = torch.argmax(predictions, dim=2)
        # argmax decoding introduces a lot of repetition

        batch_seq = list()
        for predicted_indices in batch_predicted_indices:
            s = ''
            for predicted_index in predicted_indices:
                predicted_text = self.tokenizer.decode([predicted_index])
                s += predicted_text
            batch_seq.append(s)
        return batch_seq


if __name__ == '__main__':
    # sentence = 'Jim Henson was a puppeteer'
    # bert_util = BertUtil()
    # word_embeddings = bert_util.generate_word_embeddings(sentence)
    # print('word_embeddings.shape', word_embeddings.shape)

    # sentiment_analysis_util = SentimentAnalysisUtil()
    # sentence = 'Sad'
    # vec = sentiment_analysis_util.get_sentiment_vector(sentence)
    # print(vec)

    # inputs_embeds = torch.rand((config.batch_size, config.max_length, config.gpt2_dim)).to(config.device)
    # gpt2_util = GPT2Util()
    # for sentence in gpt2_util.batch_generate_sentence(inputs_embeds):
    #     print(sentence)