import pandas as pd
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils import BertUtil, SentimentAnalysisUtil

class IMDBDataset(Dataset):
    def __init__(self, bert_util, sentiment_analysis_util, path=config.path, device=config.device):
        self.device = device 
        self.data_frame = pd.read_csv(path)
        self.clean_dataset()
        self.bert_util = bert_util
        self.sentiment_analysis_util = sentiment_analysis_util

    def clean_dataset(self):
        self.data_frame.review = self.data_frame.review.apply(lambda x: x.replace('<br />', '')[1:-1])

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        sentence = self.data_frame.review.iloc[idx]
        cls_embedding = self.bert_util.generate_cls_embedding(sentence)
        sentiment_label = self.data_frame.sentiment.iloc[idx].upper() # global dict has enum for upper-case sentiment labels
        sentiment_embedding = sentiment_analysis_util.get_sentiment_vector_from_label(sentiment_label)
        return cls_embedding, sentiment_embedding

if __name__ == "__main__":
    bert_util = BertUtil()
    sentiment_analysis_util = SentimentAnalysisUtil()
    
    dataset = IMDBDataset(bert_util=bert_util, sentiment_analysis_util=sentiment_analysis_util)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    
    cls_embedding, sentiment_embedding = next(iter(dataloader))
    
    print(cls_embedding.shape)
    # shape(cls_embedding) = [batch_size, 1, hidden_dim]
    print(sentiment_embedding.shape)
    # shape(sentiment_embedding) = [batch_size, 1, len(config.SENTIMENTS)] = [batch_size, 1, 2]
