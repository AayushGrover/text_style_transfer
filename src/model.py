import torch 
import torch.nn as nn 

import config

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.input_dim = len(config.SENTIMENTS) + config.bert_dim
        self.hidden1 = nn.Linear(self.input_dim, config.gpt2_dim)
    
    def forward(self, cls_embedding, word_embeddings, sentiment_embedding):
        # introduce a dimension along seq_len axis
        # repeat the sentiment_embedding across all seq_len dimension (make a copy for each word)
        sentiment_embedding = sentiment_embedding.unsqueeze(1).repeat(1, config.max_length, 1)
        # concatenate the same shape matrices
        conc = torch.cat((word_embeddings, sentiment_embedding), dim=2)
        latent = self.relu(self.hidden1(conc))
        return latent


if __name__ == "__main__":
    model = Net()
    model.to(config.device)

    from torch.utils.data import Dataset, DataLoader
    from utils import BertUtil, GPT2Util, SentimentAnalysisUtil
    from data_loader import IMDBDataset
    bert_util = BertUtil()
    sentiment_analysis_util = SentimentAnalysisUtil()
    dataset = IMDBDataset(bert_util=bert_util, sentiment_analysis_util=sentiment_analysis_util)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    cls_embedding, word_embeddings, sentiment_embedding = next(iter(dataloader))

    input_embeds = model(cls_embedding, word_embeddings, sentiment_embedding)

    gpt2_util = GPT2Util()
    for sentence in gpt2_util.batch_generate_sentence(input_embeds):
        output_cls_embeddings = bert_util.generate_cls_embedding(sentence)
        output_word_embeddings = bert_util.generate_word_embeddings(sentence)
        output_sentiment_embedding = sentiment_analysis_util.get_sentiment_vector(sentence)