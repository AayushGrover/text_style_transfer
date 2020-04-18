import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '../data/IMDB Dataset.csv'
train_path = '../data/train_IMDB Dataset.csv'
test_path = '../data/test_IMDB Dataset.csv'
bert_pretrained_weights = 'bert-base-uncased'
gpt2_pretrained_weights = 'gpt2'
SENTIMENTS = {'POSITIVE': 0, 'NEGATIVE': 1} # setup enumeration for both the sentiments
train = True

epochs = 10
ckpt_num = 5
batch_size = 4
max_length = 512
bert_dim = 768
gpt2_dim = 768

# for interpolating the two losses
loss_interpolation_factor_initial = 0.9
loss_interpolation_step = 0.1
loss_interpolation_limit = 0.5