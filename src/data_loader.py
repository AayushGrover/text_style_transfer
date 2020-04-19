import pandas as pd
import numpy as np 

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
        word_embeddings = self.bert_util.generate_word_embeddings(sentence)
        # sentiment_embedding = self.sentiment_analysis_util.get_sentiment_vector(sentence)
        sentiment_embedding = self.sentiment_analysis_util.get_rand_target_sentiment()
        return cls_embedding, word_embeddings, sentiment_embedding

if __name__ == '__main__':
    bert_util = BertUtil()
    sentiment_analysis_util = SentimentAnalysisUtil()
    
    dataset = IMDBDataset(bert_util=bert_util, sentiment_analysis_util=sentiment_analysis_util)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    
    cls_embedding, word_embeddings, sentiment_embedding = next(iter(dataloader))
    
    print('cls_embedding.shape', cls_embedding.shape)
    # shape(cls_embedding) = [batch_size, hidden_dim]
    print('word_embeddings.shape', word_embeddings.shape)
    # shape(word_embeddings) = [batch_size, seq_len, hidden_dim]
    print('sentiment_embedding.shape', sentiment_embedding.shape)
    # shape(sentiment_embedding) = [batch_size, len(config.SENTIMENTS)] = [batch_size, 2]
