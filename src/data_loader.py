import pandas as pd
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize

import pickle

import config
from utils import BertUtil, SentenceBERTUtil

class IMDBDataset(Dataset):
    def __init__(self, path=config.path, device=config.device):
        self.device = device 
        self.data_frame = pd.read_csv(path)
        self.clean_dataset()
        self.word2idx = pickle.load(open(f'{config.glove_path}/6B.{config.glove_embed_dim}_word2idx.pkl', 'rb'))
        # self.sentence_bert_util = SentenceBERTUtil()

    def clean_dataset(self):
        self.data_frame.review = self.data_frame.review.apply(lambda x: x.lower().replace('<br />', '')[1:-1])

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        sentence = self.data_frame.review.iloc[idx]
        # if(config.use_bert_cls_embedding == True):
        #     sentence_embedding = self.bert_util.generate_cls_embedding(sentence)
        # elif(config.use_bert_sentence_embedding == True):
        # sentence_embedding = self.bert_util.generate_sentence_embedding(sentence)
        # elif(config.use_sentence_bert_embedding == True):
        # sentence_embedding = self.sentence_bert_util.generate_sentence_embedding(sentence)
        # word_embeddings = self.bert_util.generate_word_embeddings(sentence)
        # sentiment_embedding = self.sentiment_analysis_util.get_rand_target_sentiment()
        # sentiment_embedding = self.sentiment_analysis_util.get_const_positive_sentiment()
        tokens = [0]
        for word in word_tokenize(sentence):
            tokens.append(self.word2idx[word])
            if (len(tokens) == (config.max_length - 1)):
                break
        tokens.append(1)
        for _ in range(config.max_length-len(tokens)):
            tokens.append(2)
        token_array = np.array(tokens)
        token_array = torch.from_numpy(token_array).to(device=config.device).view(-1,1)
        return token_array

if __name__ == '__main__':
    # bert_util = BertUtil()
    # sentence_bert_util = SentenceBERTUtil()
    # sentiment_analysis_util = SentimentAnalysisUtil()
    
    dataset = IMDBDataset(path=config.train_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    
    word_embeddings = next(iter(dataloader))
    
    # print('sentence_embedding.shape', sentence_embedding.shape)
    # shape(sentence_embedding) = [batch_size, hidden_dim]
    print('word_embeddings.shape', word_embeddings.shape)
    # shape(word_embeddings) = [batch_size, seq_len, hidden_dim]
    # print('sentiment_embedding.shape', sentiment_embedding.shape)
    # shape(sentiment_embedding) = [batch_size, len(config.SENTIMENTS)] = [batch_size, 2]
