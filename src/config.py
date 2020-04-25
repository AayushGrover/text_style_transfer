import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
path = '../data/IMDB Dataset.csv'
train_path = '../data/train_IMDB Dataset.csv'
test_path = '../data/test_IMDB Dataset.csv'
bert_pretrained_weights = 'bert-base-uncased'
sentence_bert_pretrained = 'bert-base-nli-stsb-mean-tokens' # suited for semantic textual similarity
gpt2_pretrained_weights = 'gpt2'
SENTIMENTS = {'POSITIVE': 0, 'NEGATIVE': 1} # setup enumeration for both the sentiments
train = True

# pick one of the following sentence embedding representations
# change the model name in the model_save_path variable
use_bert_cls_embedding = False
use_bert_sentence_embedding = False
use_sentence_bert_embedding = True
assert([use_bert_cls_embedding, use_bert_sentence_embedding, use_sentence_bert_embedding].count(True) == 1)

epochs = 10
ckpt_num = 1
model_save_path = '../models/sentence_bert_factor_initial(0.9)_step(0.1)_limit(0.5).pt'
batch_size = 4
max_length = 512
bert_dim = 768
gpt2_dim = 768

# for interpolating the two losses
# change the model name in the model_save_path variable
loss_interpolation_factor_initial = 0.9
loss_interpolation_step = 0.1
loss_interpolation_limit = 0.5

semantic_meaning_weight = 1
sentiment_weight = 0