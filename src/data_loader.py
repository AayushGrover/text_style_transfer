import pandas as pd
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

import config

class IMDBDataset(Dataset):
    def __init__(self, path=config.path, max_length=config.max_length, pretrained_weights=config.pretrained_weights, device=config.device):
        self.device = device 
        self.max_length = max_length
        self.data_frame = pd.read_csv(path)
        self.clean_dataset()
        self.pretrained_weights = pretrained_weights
        self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.tokenizer = Tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)

    def clean_dataset(self):
        self.data_frame.review = self.data_frame.review.apply(lambda x: x.replace('<br />', '')[1:-1])

    def _generate_input_ids(self, sentence):
        encoded_dict = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_length, pad_to_max_length=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids'].squeeze(0)
        # shape(input_ids) = [batch_size, max_length]
        return input_ids

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        sentence = self.data_frame.review.iloc[idx]
        return torch.tensor(self._generate_input_ids(sentence), device=self.device)

if __name__ == "__main__":
    dataset = IMDBDataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    print(next(iter(dataloader)).shape)
