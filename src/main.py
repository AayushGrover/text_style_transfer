import os 
import time
from tqdm import tqdm 

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils import BertUtil, SentenceBERTUtil, GPT2Util, SentimentAnalysisUtil
from data_loader import IMDBDataset
from model import Net
from losses import loss_semantic_meaning, loss_sentiment
import config
import torch

def train(model, 
        train_dl,
        bert_util, 
        sentiment_analysis_util, 
        gpt2_util, 
        optimizer, 
        epochs=config.epochs):

    writer = SummaryWriter()

    alpha = config.loss_interpolation_factor_initial
    alpha_step = config.loss_interpolation_step
    alpha_min = config.loss_interpolation_limit

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch}')
        epoch_train_loss = 0
        if(epoch != 1 and ((alpha - alpha_step) >= alpha_min)):
            alpha = alpha - alpha_step
        
        for input_sentence_embedding, input_word_embeddings, input_sentiment_embeddings in tqdm(train_dl):
            gpt2_input_embeds = model(input_sentence_embedding, input_word_embeddings, input_sentiment_embeddings)
            
            sentences = gpt2_util.batch_generate_sentence(gpt2_input_embeds)
            if(config.use_bert_cls_embedding == True):
                output_batch_sentence_embedding = bert_util.generate_batch_cls_embeddings(sentences)
            elif(config.use_bert_sentence_embedding == True):
                output_batch_sentence_embedding = bert_util.generate_batch_sentence_embedding(sentences)
            elif(config.use_sentence_bert_embedding == True):
                output_batch_sentence_embedding = sentence_bert_util.generate_batch_sentence_embedding(sentences)
            output_sentiment_embedding_batch = sentiment_analysis_util.get_batch_sentiment_vectors(sentences)

            # input sentiment embeddings are going to be used as the targets for the loss
            semantic_meaning_loss = (alpha) * config.semantic_meaning_weight * loss_semantic_meaning(input_sentence_embedding, output_batch_sentence_embedding)
            sentiment_loss = (1 - alpha) * config.sentiment_weight * loss_sentiment(input_sentiment_embeddings, output_sentiment_embedding_batch)
            loss = torch.sum(semantic_meaning_loss+sentiment_loss, dim=0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        print(f'Loss: {epoch_train_loss}')
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)

        if(epoch % config.ckpt_num == 0):
            torch.save(model.state_dict(), config.model_save_path)

    writer.close()

def test(model,
          test_dl,
          gpt2_util):

    for i, (input_sentence_embedding, input_word_embeddings, input_sentiment_embeddings) in enumerate(test_dl):
        gpt2_input_embeds = model(input_sentence_embedding, input_word_embeddings, input_sentiment_embeddings)
        sentences = gpt2_util.batch_generate_sentence(gpt2_input_embeds)

        print("Batch Number:", i+1)
        print("-"*20)
        print("Input sentiment embeddings:")
        print(input_sentiment_embeddings)
        print("-"*20)
        print("Produced sentences:\n")
        print(sentences)
        print("-"*20)
        input()


if __name__ == '__main__':
    bert_util = BertUtil()
    gpt2_util = GPT2Util()
    sentence_bert_util = SentenceBERTUtil()
    sentiment_analysis_util = SentimentAnalysisUtil()
    
    train_dataset = IMDBDataset(bert_util=bert_util, 
                            sentence_bert_util=sentence_bert_util,
                            sentiment_analysis_util=sentiment_analysis_util, 
                            path=config.train_path)
    test_dataset = IMDBDataset(bert_util=bert_util, 
                            sentence_bert_util=sentence_bert_util,
                            sentiment_analysis_util=sentiment_analysis_util, 
                            path=config.test_path)
    
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size)
    test_dl = DataLoader(test_dataset, batch_size=config.batch_size)

    model = Net()

    if(config.train == True):
        model.to(config.device)

        print('Total number of parameters', sum(p.numel() for p in model.parameters()))
        print('Number of trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = optim.Adam(model.parameters())

        train(model=model, 
            train_dl=train_dl, 
            bert_util=bert_util, 
            sentiment_analysis_util=sentiment_analysis_util, 
            gpt2_util=gpt2_util, 
            optimizer=optimizer)
    else:
        # model = torch.load_state_dict(torch.load(config.model_save_path))
        model.to(config.device)

        test(model=model,
              test_dl=test_dl,
              gpt2_util=gpt2_util)
