import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
from nltk.tokenize import word_tokenize

import random
import pickle

import config

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.load(f'{config.glove_path}/glove_embeddings_{config.glove_embed_dim}.pt').to(config.device)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden, batch_size=config.batch_size):
        '''
        input = [max_len, batch_size]
        '''
        embedded = self.embedding(input).view(1,batch_size,self.hidden_size)
        output = embedded
        # print(output.shape)
        # quit()
        output, hidden = self.gru(output.float(), hidden)
        
        return output, hidden

    def initHidden(self, batch_size=config.batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=config.device)



class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.load(f'{config.glove_path}/glove_embeddings_{config.glove_embed_dim}.pt').to(config.device)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, batch_size=config.batch_size):
        output = self.embedding(input).view(1,batch_size,self.hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output.float(), hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size=config.batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=config.device)

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.encoder = Encoder(input_size=input_size, hidden_size=config.glove_embed_dim)
        self.decoder = Decoder(hidden_size=config.glove_embed_dim, output_size=output_size)
        self.device = config.device
        
        assert self.encoder.hidden_size == self.decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, input, batch_size=config.batch_size, max_length=config.max_length):
        '''
        input = input tensor
        '''
        encoder_hidden = self.encoder.initHidden(batch_size)

        input_length = max_length
        target_length = max_length

        encoder_outputs = torch.zeros(max_length, batch_size, self.encoder.hidden_size, device=self.device)

        # print(input_length)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = torch.tensor([[0]*batch_size], device=self.device)

        decoder_hidden = encoder_hidden

        output = torch.zeros(max_length, batch_size)
        
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden= self.decoder(
                decoder_input, decoder_hidden)
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