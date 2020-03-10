import pandas as pd
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader

import config

class IMDBDataset(Dataset):
    def __init__(self, path=config.path, device=config.device):
        self.device = device
        self.data_frame = pd.read_csv(path)
    
    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(np.array(self.data_frame.iloc[idx, :]), device=self.device)

if __name__ == "__main__":
    dataset = IMDBDataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    print(next(iter(dataloader)))