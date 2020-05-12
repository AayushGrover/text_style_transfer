import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
path = '../data/IMDB_Dataset.csv'
train_path = '../data/train_IMDB Dataset.csv'
test_path = '../data/test_IMDB Dataset.csv'
bert_pretrained_weights = 'bert-base-uncased'
sentence_bert_pretrained = 'bert-base-nli-stsb-mean-tokens' # suited for semantic textual similarity
glove_path = '../embeddings/glove.6B'
SENTIMENTS = {'POSITIVE': 0, 'NEGATIVE': 1} # setup enumeration for both the sentiments
train = True

epochs = 10
# dropout = 0.8
num_layers = 1
hidden_size = 64
ckpt_num = 1
# model_save_path = '../models/sentence_bert_factor_initial(0.9)_step(0.1)_limit(0.5).pt'
model_save_path = '../models/seq2seq_factor_initial(0.9)_step(0.1)_limit(0.5).pt'
batch_size = 4
max_length = 512
bert_dim = 768
glove_embed_dim = 100 # can be one of [50, 100, 200, 300]

# for interpolating the two losses
# change the model name in the model_save_path variable
loss_interpolation_factor_initial = 0.9
loss_interpolation_step = 0.1
loss_interpolation_limit = 0.5

semantic_meaning_weight = 1
sentiment_weight = 0

