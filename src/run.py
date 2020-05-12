import os 
import time
from tqdm import tqdm 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np

import pickle

import config
from model import Seq2Seq
from data_loader import IMDBDataset
from utils import SentenceBERTUtil, SentimentAnalysisUtil
from losses import loss_sentiment, loss_semantic

def train(model, train_dl, optimizer, epochs=config.epochs):
    writer = SummaryWriter()
    sent_analyser = SentimentAnalysisUtil()
    sentence_bert = SentenceBERTUtil()

    alpha = config.loss_interpolation_factor_initial
    alpha_step = config.loss_interpolation_step
    alpha_min = config.loss_interpolation_limit

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch}')
        epoch_train_loss = 0
        if (epoch != 1 and ((alpha - alpha_step) >= alpha_min)):
            alpha = alpha - alpha_step
        for input_token_seq in tqdm(train_dl):
            
            input_token_seq = input_token_seq.squeeze() #[batch_size, max_len]
            input_sentiment_embeddings = sent_analyser.get_target_sentiment_vectors(input_token_seq)
            input_token_seq = input_token_seq.squeeze().t() #[max_len, batch_size]

            output_seq = model(input_token_seq, input_sentiment_embeddings) #[max_len, batch_size]

            input_token_seq = input_token_seq.t() #[batch_size, max_len]
            output_seq = output_seq.t() #[batch_size, max_len]

            input_token_seq = input_token_seq.cpu().numpy().tolist()
            output_seq = output_seq.cpu().numpy().tolist()
            

            semantic_meaning_loss = (alpha) * loss_semantic(input_token_seq, output_seq,sentence_bert)
            sentiment_loss = (1 - alpha) * loss_sentiment(input_token_seq, output_seq,sent_analyser)
            loss = torch.sum(semantic_meaning_loss+sentiment_loss, dim=0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        print(f'Loss: {epoch_train_loss}')
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)

        if(epoch % config.ckpt_num == 0):
            print('Saving model')
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, config.model_save_path)

    writer.close()

def test(model, test_dl):
    for i, input_token_seq in enumerate(test_dl):
        input_token_seq = input_token_seq.squeeze().t() #[max_len, batch_size]
        output_seq = model(input_token_seq) #[max_len, batch_size]

        # print("Batch Number:", i+1)
        # print("-"*20)
        # print("Input sentiment embeddings:")
        # print(input_sentiment_embeddings)
        # print("-"*20)
        # print("Produced sentences:\n")
        # print(sentences)
        # print("-"*20)
        # input()

if __name__ == '__main__':
    train_dataset = IMDBDataset(path=config.train_path)
    test_dataset = IMDBDataset(path=config.test_path)
    
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size)
    test_dl = DataLoader(test_dataset, batch_size=config.batch_size)

    model = Seq2Seq(input_size=config.max_length, output_size=config.max_length)

    if(config.train == True):
        try: 
            checkpoint = torch.load(config.model_save_path)
            model.load_state_dict(checkpoint['model'])
            print('Model loaded.')
        except:
            pass

        model.to(config.device)
        model.train()

        print('Total number of parameters', sum(p.numel() for p in model.parameters()))
        print('Number of trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = optim.Adam(model.parameters())
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Optimizer loaded.')
        except:
            pass

        train(model=model, 
            train_dl=train_dl,
            optimizer=optimizer)
    else:
        #model.load_state_dict(torch.load(config.model_save_path))
        model.to(config.device)
        model.eval()
        test(model=model,
              test_dl=test_dl)
