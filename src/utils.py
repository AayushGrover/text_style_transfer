import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer

import pickle

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
            out = self.model(input_ids)[0]  # disable grad for BERT model
        cls_embedding = out[:, 0, :].squeeze(0).to(config.device)
        # shape(cls_embedding) = [1, hidden_dim]
        cls_embedding.requires_grad = True  # enable grad (finetuning) for the vector obtained from the BERT model
        return cls_embedding
    
    def _generate_batch_cls_embeddings(self, batch_input_ids):
        with torch.no_grad():
            out = self.model(batch_input_ids)[0] # disable grad for BERT model
        batch_cls_embeddings = out[:, 0, :].to(config.device)
        # shape(batch_cls_embeddings) = [batch_size, hidden_dim]
        batch_cls_embeddings.requires_grad = True # enable grad (finetuning) for the vector obtained from the BERT model
        return batch_cls_embeddings
    
    def _generate_word_embeddings(self, input_ids):
        with torch.no_grad():
            out = self.model(input_ids)[0].squeeze(0).to(config.device) # disable grad for BERT model 
        # shape(out) = [seq_len, hidden_dim]
        out.requires_grad = True    # enable grad (finetuning) for the vector obtained from the BERT model
        return out
    
    def _generate_batch_word_embeddings(self, batch_input_ids):
        with torch.no_grad():
            out = self.model(batch_input_ids)[0].to(config.device)
        # shape(out) = [batch_size, seq_len, hidden_dim]
        out.requires_grad = True    # enable grad (finetuning) for the vector obtained from the BERT model
        return out

    def _generate_sentence_embedding(self, input_ids):
        with torch.no_grad():
            out = self.model(input_ids)[0].squeeze(0).to(config.device) # disable grad for BERT model 
        out = torch.sum(out[1:], dim=0) # sum all the word piece tokens in the seq (apart from the starting [CLS] token)
        # shape(out) = [hidden_dim]
        out.requires_grad = True    # enable grad (finetuning) for the vector obtained from the BERT model
        return out
    
    def _generate_batch_sentence_embedding(self, batch_input_ids):
        with torch.no_grad():
            out = self.model(batch_input_ids)[0].to(config.device)
        out = torch.sum(out[:, 1:, :], dim=1)   # sum all the word piece tokens in the seq (apart from the starting [CLS] token)
        # shape(out) = [batch_size, hidden_dim]
        out.requires_grad = True    # enable grad (finetuning) for the vector obtained from the BERT model
        return out
    
    def generate_cls_embedding(self, sentence):
        input_ids = self._generate_input_ids(sentence)
        cls_embedding = self._generate_cls_embedding(input_ids)
        return cls_embedding
    
    def generate_word_embeddings(self, sentence):
        input_ids = self._generate_input_ids(sentence)
        word_embeddings = self._generate_word_embeddings(input_ids)
        return word_embeddings

    def generate_sentence_embedding(self, sentence):
        input_ids = self._generate_input_ids(sentence)
        sentence_embedding = self._generate_sentence_embedding(input_ids)
        return sentence_embedding

    def generate_batch_cls_embeddings(self, batch_sentences):
        l = list()
        for sentence in batch_sentences:
            l.append(self._generate_input_ids(sentence).squeeze(0))
        batch_input_ids = torch.stack(l)
        return self._generate_batch_cls_embeddings(batch_input_ids)
    
    def generate_batch_word_embeddings(self, batch_sentences):
        l = list()
        for sentence in batch_sentences:
            l.append(self._generate_input_ids(sentence).squeeze(0))
        batch_input_ids = torch.stack(l)
        return self._generate_batch_word_embeddings(batch_input_ids)
    
    def generate_batch_sentence_embedding(self, batch_sentences):
        l = list()
        for sentence in batch_sentences:
            l.append(self._generate_input_ids(sentence).squeeze(0))
        batch_input_ids = torch.stack(l)
        return self._generate_batch_sentence_embedding(batch_input_ids)


class SentenceBERTUtil():
    def __init__(self, pretrained_weights=config.sentence_bert_pretrained):
        self.pretrained_weights = pretrained_weights
        self.model = SentenceTransformer(self.pretrained_weights, device=config.device)
        self.model.eval()
        self.idx2word = pickle.load(open(f'{config.glove_path}/6B.{config.glove_embed_dim}_idx2word.pkl', 'rb'))
    
    def _get_sentence(self, tokens):
        sentences = list()
        for batch in tokens:        
            s = ""
            for token in batch:
                s += self.idx2word[token]+" "
            
            sentences.append(s.strip())
        return sentences

    def generate_sentence_embedding(self, sentence):
        # sentence = self._get_sentence(tokens)
        assert(type(sentence) == str)
        with torch.no_grad():
            sentence_embedding = np.array(self.model.encode([sentence], show_progress_bar=False))
        sentence_embedding = torch.from_numpy(sentence_embedding).squeeze(0).to(config.device)
        # shape(sentence_embedding) = [hidden_dim]
        sentence_embedding.requires_grad = True    # enable grad (finetuning) for the vector obtained from the Sentence-BERT model
        return sentence_embedding

    def generate_batch_sentence_embedding(self, tokens):
        batch_sentences = self._get_sentence(tokens)
        with torch.no_grad():
            batch_sentence_embeddings = np.array(self.model.encode(batch_sentences, show_progress_bar=False))
        batch_sentence_embeddings = torch.from_numpy(batch_sentence_embeddings).to(config.device)
        # shape(batch_sentence_embeddings) = [batch_size, hidden_dim]
        batch_sentence_embeddings.requires_grad = True    # enable grad (finetuning) for the vector obtained from the Sentence-BERT model
        return batch_sentence_embeddings

class SentimentAnalysisUtil():
    def __init__(self, SENTIMENTS=config.SENTIMENTS):
        # leverages a fine-tuned model on sst2, which is a GLUE task.
        self.idx2word = pickle.load(open(f'{config.glove_path}/6B.{config.glove_embed_dim}_idx2word.pkl', 'rb'))
        self.nlp = pipeline('sentiment-analysis')
        self.SENTIMENTS = SENTIMENTS
    
    def _get_sentiment_label(self, sentence):
        sentence = sentence[:512]
        result = self.nlp(sentence)
        sentiment_label = result[0]['label']
        return sentiment_label
    
    def _get_sentiment_vector(self, sentiment_label):
        vec = torch.zeros(len(self.SENTIMENTS), dtype=torch.float, device=config.device)
        vec[self.SENTIMENTS[sentiment_label]] = 1
        # shape(vec) = [len(self.SENTIMENTS)] = [2]
        return vec
    
    def _get_sentence(self, tokens):
        sentences = list()
        for batch in tokens:        
            s = ""
            for token in batch:
                try: 
                    s += self.idx2word[token.item()]+" "
                except:
                    s += self.idx2word[token]+" "

            sentences.append(s.strip())
        return sentences

    def get_sentiment_vector(self, sentence):
        # sentence = self._get_sentence(tokens)
        sentiment_label = self._get_sentiment_label(sentence)
        vec = self._get_sentiment_vector(sentiment_label)
        return vec
    
    def get_flipped_sentiment_vector(self, sentence):
        # sentence = self._get_sentence(tokens)
        sentiment_label = self._get_sentiment_label(sentence)
        if sentiment_label == "NEGATIVE":
            sentiment_label = "POSITIVE"
        else:
            sentiment_label = "NEGATIVE"
        vec = self._get_sentiment_vector(sentiment_label)
        return vec
    
    def get_batch_sentiment_vectors(self, tokens):
        sentences = self._get_sentence(tokens)
        l = list()
        for sentence in sentences:
            l.append(self.get_sentiment_vector(sentence))
        vectors = torch.stack(l)
        return vectors

    def get_target_sentiment_vectors(self, tokens):
        sentences = self._get_sentence(tokens)
        l = list()
        for sentence in sentences:
            l.append(self.get_flipped_sentiment_vector(sentence))
        vectors = torch.stack(l)
        return vectors

#     def get_sentiment_vector_from_label(self, sentiment_label):
#         return self._get_sentiment_vector(sentiment_label)

#     def get_rand_target_sentiment(self):
#         target_sentiment = np.random.choice(list(self.SENTIMENTS)) 
#         return self._get_sentiment_vector(target_sentiment)
    
#     def get_const_positive_sentiment(self):
#         positive_str = 'POSITIVE'
#         assert(positive_str in self.SENTIMENTS)
#         return self._get_sentiment_vector(positive_str)


# class GPT2Util():
#     def __init__(self, pretrained_weights=config.gpt2_pretrained_weights, max_length=config.max_length):
#         self.pretrained_weights = pretrained_weights
#         self.max_length = max_length
#         self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_weights)
#         self.model.to(config.device)
#         self.model.eval()
#         self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_weights)
    
#     def batch_generate_sentence(self, inputs_embeds):
#         with torch.no_grad():
#             predictions = self.model(inputs_embeds=inputs_embeds)[0]
#             # shape(predictions) = [batch_size, max_length, gpt2_vocab_size]

#         batch_predicted_indices = torch.argmax(predictions, dim=2)
#         # argmax decoding introduces a lot of repetition

#         batch_seq = list()
#         for predicted_indices in batch_predicted_indices:
#             s = ''
#             for predicted_index in predicted_indices:
#                 predicted_text = self.tokenizer.decode([predicted_index])
#                 s += predicted_text
#             batch_seq.append(s)
#         return batch_seq


def generate_train_test_split(path=config.path, train_path=config.train_path, test_path=config.test_path):
    d = pd.read_csv(path)
    review = d.review
    sentiment = d.sentiment
    review_train, review_test, sentiment_train, sentiment_test = train_test_split(review, sentiment, test_size=0.3, random_state=42)
    train_d = pd.DataFrame({'review': review_train, 'sentiment': sentiment_train})
    test_d = pd.DataFrame({'review': review_test, 'sentiment': sentiment_test})
    train_d.to_csv(train_path, index=False)
    test_d.to_csv(test_path, index=False)


if __name__ == '__main__':
    bert_util = BertUtil()
    sentence_bert_util = SentenceBERTUtil()

    sentence = 'Jim Henson was a puppeteer'

    sentence_embedding = bert_util.generate_cls_embedding(sentence)
    print('bert_cls_embedding.shape', sentence_embedding.shape)
    sentence_embedding = bert_util.generate_sentence_embedding(sentence)
    print('bert_sentence_embedding.shape', sentence_embedding.shape)

    sentence_embedding = sentence_bert_util.generate_sentence_embedding(sentence)
    print('sentence_bert_embedding.shape', sentence_embedding.shape)

    # sentiment_analysis_util = SentimentAnalysisUtil()
    # sentence = 'Sad'
    # vec = sentiment_analysis_util.get_sentiment_vector(sentence)
    # print(vec)

    # inputs_embeds = torch.rand((config.batch_size, config.max_length, config.gpt2_dim)).to(config.device)
    # gpt2_util = GPT2Util()
    # print(gpt2_util.batch_generate_sentence(inputs_embeds))

    # generate_train_test_split()