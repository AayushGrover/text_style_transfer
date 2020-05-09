import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn

import pickle

import config

class GloVeEmbed():
    def __init__(self):
        self.glove_path = config.glove_path
        self.words = list()
        self.idx = 0
        self.word2idx = dict()
        self.idx2word = dict()
        self.vec_dim = config.glove_embed_dim
        # self.vectors = bcolz.carray(np.zeros(1), rootdir=f'{self.glove_path}/6B.{self.vec_dim}.dat', mode='w')
        self.vectors = list()

    def _store_vectors(self):
        with open(f'{self.glove_path}/glove.6B.{self.vec_dim}d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                self.words.append(word)
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word 
                self.idx += 1
                vect = np.array(line[1:]).astype(np.float)
                self.vectors.append(vect)
    
        # vectors = bcolz.carray(self.vectors[1:].reshape((400000, self.vec_dim)), rootdir=f'{self.glove_path}/6B.{self.vec_dim}.dat', mode='w')
        # vectors.flush()
        # pickle.dump(self.words, open(f'{self.glove_path}/6B.{self.vec_dim}_words.pkl', 'wb'))
                
        self.vectors = np.array(self.vectors)
        self.vectors = self.vectors.reshape(400000, self.vec_dim)
        return self.vectors, self.words

    def _get_target_vocab(self):
        d = pd.read_csv(config.path)
        d.review = d.review.apply(lambda x: x.replace('<br />', '')[1:-1])

        target_vocab = set()
        target_vocab.add('<SOS>') 
        target_vocab.add('<EOS>') 
        target_vocab.add('<PAD>')
        target_vocab.add('positive') 
        target_vocab.add('negative') 
        
        for line in d.review:
            for word in word_tokenize(line):
                target_vocab.add(word)
        
        target_vocab = list(target_vocab)
        return target_vocab

    def _get_matrix(self, glove, target_vocab):
        self.idx = 0
        self.idx2word = dict()
        self.word2idx = dict()
        matrix_len = len(target_vocab)
        weights_matrix = np.zeros((matrix_len, self.vec_dim))
        words_found = 0

        for i, word in enumerate(target_vocab):
            try: 
                weights_matrix[i] = glove[word]
                words_found += 1
                
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.vec_dim, )) 
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word 
            self.idx += 1 
        emb_layer = nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix))
        
        pickle.dump(self.word2idx, open(f'{self.glove_path}/6B.{self.vec_dim}_word2idx.pkl', 'wb'))
        pickle.dump(self.idx2word, open(f'{self.glove_path}/6B.{self.vec_dim}_idx2word.pkl', 'wb'))

        print(matrix_len, words_found, len(self.word2idx))
        return emb_layer

    def get_embeddings(self):
        
        vectors, words = self._store_vectors()
        glove = {w: vectors[self.word2idx[w]] for w in words}
        target_vocab = self._get_target_vocab()

        emb_layer = self._get_matrix(glove, target_vocab)
        torch.save(emb_layer, f'{self.glove_path}/glove_embeddings_{self.vec_dim}.pt')

if __name__ == "__main__":

    GloVeEmbed().get_embeddings()