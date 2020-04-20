import config
from utils import BertUtil

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel

'''
Used to compute and store the BERT embeddings for the entire dataset.
'''

class IMDBDataset(Dataset):
    def __init__(self, bert_util, path=config.path, device=config.device):
        self.device = device 
        self.data_frame = pd.read_csv(path)
        self.clean_dataset()
        self.bert_util = bert_util

    def clean_dataset(self):
        self.data_frame.review = self.data_frame.review.apply(lambda x: x.replace('<br />', '')[1:-1])

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        sentence = self.data_frame.review.iloc[idx]
        cls_embedding = self.bert_util.generate_cls_embedding(sentence)
        word_embeddings = self.bert_util.generate_word_embeddings(sentence)

        return cls_embedding, word_embeddings


if __name__ == "__main__":
    bert = BertUtil()

    data = IMDBDataset(bert)
    dataloader = DataLoader(data, batch_size=config.batch_size)
    
    # df1 = pd.read_csv("../data/test_IMDB Dataset.csv")
    # df2 = pd.read_csv("../data/train_IMDB Dataset.csv")
    
    # df = pd.concat([df1, df2], ignore_index=True)

    # df.to_csv("../data/IMDB_Dataset.csv", index=False)

    #cls - [4, 768] ; word - [4, 512, 768]

    cls_all = list()
    word_all = list()

    for cls_embedding, word_embeddings in iter(dataloader):
        cls_all.append(cls_embedding)
        word_all.append(word_embeddings)

    cls_stack = torch.stack(cls_all)
    word_stack = torch.stack(word_all)

    torch.save(cls_stack, "../embeddings/cls.pth")
    torch.save(word_stack, "../embeddings/word.pth")

    print(cls_stack.shape)
    print(word_stack.shape)