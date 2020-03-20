import torch

batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '../data/IMDB Dataset.csv'
pretrained_weights = 'bert-base-uncased'
max_length = 512