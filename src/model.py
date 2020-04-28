import torch 
import torch.nn as nn 

import config

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.input_dim = config.bert_dim
        self.hidden1 = nn.Linear(self.input_dim, config.gpt2_dim)
    
    def forward(self, cls_embedding, word_embeddings, sentiment_embedding):
        # introduce a dimension along seq_len axis
        # repeat the sentiment_embedding across the seq_len dimension (make a copy for each word)
        # sentiment_embedding = sentiment_embedding.unsqueeze(1).repeat(1, config.max_length, 1)
        # concatenate the same shape matrices
        # conc = torch.cat((word_embeddings, sentiment_embedding), dim=2)
        conc = word_embeddings
        latent = self.relu(self.hidden1(conc))
        return latent


if __name__ == '__main__':
    model = Net()
    model.to(config.device)

    from torch.utils.data import Dataset, DataLoader
    from utils import BertUtil, GPT2Util, SentimentAnalysisUtil
    from data_loader import IMDBDataset
    bert_util = BertUtil()
    sentiment_analysis_util = SentimentAnalysisUtil()
    gpt2_util = GPT2Util()

    train_dataset = IMDBDataset(bert_util=bert_util, 
                            sentiment_analysis_util=sentiment_analysis_util, 
                            path=config.train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    cls_embedding, word_embeddings, sentiment_embedding = next(iter(train_dataloader))

    input_embeds = model(cls_embedding, word_embeddings, sentiment_embedding)

    batch_sentences = gpt2_util.batch_generate_sentence(input_embeds)
    batch_cls_embeddings = bert_util.generate_batch_cls_embeddings(batch_sentences)
    batch_word_embeddings = bert_util.generate_batch_word_embeddings(batch_sentences)
    batch_sentiment_embeddings = sentiment_analysis_util.get_batch_sentiment_vectors(batch_sentences)