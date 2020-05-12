import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
from nltk.tokenize import word_tokenize

import random
import pickle

import config

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=config.num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = torch.load(f'{config.glove_path}/glove_embeddings_{config.glove_embed_dim}.pt').to(config.device)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True)
        # self.gru = nn.GRU(hidden_size+2, hidden_size+2, num_layers=num_layers, bidirectional=True)

    def forward(self, input, sentiment, hidden, batch_size=config.batch_size):
        '''
        input = [max_len, batch_size]
        '''
        embedded = self.embedding(input).view(1,batch_size,self.hidden_size)
        # sentiment = sentiment.view(1,batch_size,-1)
        # output = torch.cat((embedded.float(), sentiment), 2)
        output = embedded.float() # Comment this line when passing sentiment embeddings
        output, hidden = self.gru(output, hidden)
        
        return output, sentiment, hidden

    def initHidden(self, batch_size=config.batch_size):
        return torch.zeros(2*self.num_layers, batch_size, self.hidden_size, device=config.device)
        # return torch.zeros(2*self.num_layers, batch_size, self.hidden_size+2, device=config.device)



class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=config.num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = torch.load(f'{config.glove_path}/glove_embeddings_{config.glove_embed_dim}.pt').to(config.device)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True)
        # self.gru = nn.GRU(hidden_size+2, hidden_size+2, num_layers=num_layers, bidirectional=True)
        self.out = nn.Linear(2*(hidden_size), output_size)
        # self.out = nn.Linear(2*(hidden_size+2), output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, sentiment, hidden, batch_size=config.batch_size):
        output = self.embedding(input).view(1,batch_size,self.hidden_size)
        # output = torch.cat((output.float(),sentiment), 2)
        output = F.relu(output.float())
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size=config.batch_size):
        return torch.zeros(2*self.num_layers, batch_size, self.hidden_size, device=config.device)
        # return torch.zeros(2*self.num_layers, batch_size, self.hidden_size+2, device=config.device)

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.encoder = Encoder(input_size=input_size, hidden_size=config.glove_embed_dim)
        self.decoder = Decoder(hidden_size=config.glove_embed_dim, output_size=output_size)
        self.device = config.device
        
        assert self.encoder.hidden_size == self.decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, input, sentiment=None, batch_size=config.batch_size, max_length=config.max_length):
        '''
        input = input tensor
        '''
        encoder_hidden = self.encoder.initHidden(batch_size)

        input_length = max_length
        target_length = max_length

        encoder_outputs = torch.zeros(max_length, batch_size, 2*(self.encoder.hidden_size), device=self.device)
        # encoder_outputs = torch.zeros(max_length, batch_size, 2*(self.encoder.hidden_size+2), device=self.device)

        # print(input_length)
        for ei in range(input_length):
            encoder_output, encoder_sentiment, encoder_hidden = self.encoder(
                input[ei], sentiment, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = torch.tensor([[0]*batch_size], device=self.device)

        decoder_sentiment = encoder_sentiment

        decoder_hidden = encoder_hidden

        output = torch.zeros(max_length, batch_size)
        
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden= self.decoder(
                decoder_input, decoder_sentiment, decoder_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            output[di] = decoder_input
            
        
        return output
        


if __name__ == "__main__":
    inp = "this was a great movie."
    
    word2idx = pickle.load(open(f'{config.glove_path}/6B.{config.glove_embed_dim}_word2idx.pkl', 'rb'))
    idx2word = pickle.load(open(f'{config.glove_path}/6B.{config.glove_embed_dim}_idx2word.pkl', 'rb'))
    
    tokens = [0]
    for word in word_tokenize(inp):
        tokens.append(word2idx[word])
    tokens.append(1)
    for i in range(config.max_length-len(tokens)):
        tokens.append(2)
    
    token_array = np.array(tokens)

    model = Seq2Seq(input_size=config.max_length, output_size=config.max_length).to(config.device)

    out = model(torch.from_numpy(token_array).to(device=config.device).view(-1,1))
    
    s = ""
    for token in out:
        s += idx2word[token.item()] 
    print(s)
    # out = out[0].data
    # print(out.shape)
    # print([idx2word[token.item()] for token in out])