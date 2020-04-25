import pandas as pd
import numpy as np 

from torch.utils.data import Dataset, DataLoader

import config
from utils import BertUtil, SentenceBERTUtil, SentimentAnalysisUtil

class IMDBDataset(Dataset):
    def __init__(self, bert_util, sentence_bert_util, sentiment_analysis_util, path=config.path, device=config.device):
        self.device = device 
        self.data_frame = pd.read_csv(path)
        self.clean_dataset()
        self.bert_util = bert_util
        self.sentence_bert_util = sentence_bert_util
        self.sentiment_analysis_util = sentiment_analysis_util

    def clean_dataset(self):
        self.data_frame.review = self.data_frame.review.apply(lambda x: x.replace('<br />', '')[1:-1])

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        sentence = self.data_frame.review.iloc[idx]
        if(config.use_bert_cls_embedding == True):
            sentence_embedding = self.bert_util.generate_cls_embedding(sentence)
        elif(config.use_bert_sentence_embedding == True):
            sentence_embedding = self.bert_util.generate_sentence_embedding(sentence)
        elif(config.use_sentence_bert_embedding == True):
            sentence_embedding = self.sentence_bert_util.generate_sentence_embedding(sentence)
        word_embeddings = self.bert_util.generate_word_embeddings(sentence)
        # sentiment_embedding = self.sentiment_analysis_util.get_rand_target_sentiment()
        sentiment_embedding = self.sentiment_analysis_util.get_const_positive_sentiment()
        return sentence_embedding, word_embeddings, sentiment_embedding

if __name__ == '__main__':
    bert_util = BertUtil()
    sentence_bert_util = SentenceBERTUtil()
    sentiment_analysis_util = SentimentAnalysisUtil()
    
    dataset = IMDBDataset(bert_util=bert_util, 
                            sentence_bert_util=sentence_bert_util, 
                            sentiment_analysis_util=sentiment_analysis_util, 
                            path=config.train_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    
    sentence_embedding, word_embeddings, sentiment_embedding = next(iter(dataloader))
    
    print('sentence_embedding.shape', sentence_embedding.shape)
    # shape(sentence_embedding) = [batch_size, hidden_dim]
    print('word_embeddings.shape', word_embeddings.shape)
    # shape(word_embeddings) = [batch_size, seq_len, hidden_dim]
    print('sentiment_embedding.shape', sentiment_embedding.shape)
    # shape(sentiment_embedding) = [batch_size, len(config.SENTIMENTS)] = [batch_size, 2]
