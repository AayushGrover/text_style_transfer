import torch

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = '../data/IMDB Dataset.csv'
