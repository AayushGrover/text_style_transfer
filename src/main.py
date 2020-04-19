import os 
import time
from tqdm import tqdm 

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils import BertUtil, GPT2Util, SentimentAnalysisUtil
from data_loader import IMDBDataset
from model import Net
from losses import loss_semantic_meaning, loss_sentiment
import config
import torch

def train(model, 
        train_dl, 
        test_dl, 
        bert_util, 
        sentiment_analysis_util, 
        gpt2_util, 
        optimizer, 
        epochs=config.epochs, 
        device=config.device):

    writer = SummaryWriter()

    alpha = config.loss_interpolation_factor_initial
    alpha_step = config.loss_interpolation_step
    alpha_min = config.loss_interpolation_limit

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch}')
        if epoch != 1 and ((alpha - alpha_step) >= alpha_min):
            alpha = alpha - alpha_step
        
        epoch_train_loss = 0
        for input_cls_embedding,input_word_embeddings,input_sentiment_embeddings in tqdm(train_dl):
            gpt2_input_embeds = model(input_cls_embedding, input_word_embeddings, input_sentiment_embeddings)

            output_cls_batch = []
            output_sentiment_embedding_batch = []
            for sentence in gpt2_util.batch_generate_sentence(gpt2_input_embeds):
                output_cls_embeddings = bert_util.generate_cls_embedding(sentence)
                output_sentiment_embedding = sentiment_analysis_util.get_sentiment_vector(sentence)
                
                output_cls_batch.append(output_cls_embeddings)
                output_sentiment_embedding_batch.append(output_sentiment_embedding)
            
            output_cls_batch = torch.stack(output_cls_batch)
            output_sentiment_embedding_batch = torch.stack(output_sentiment_embedding_batch)

            # input sentiment embeddings are going to be used as the targets for the loss
            # print('input_cls_embedding.requires_grad', input_cls_embedding.requires_grad)
            # print('output_cls_batch.requires_grad', output_cls_batch.requires_grad)
            # print('input_sentiment_embeddings.requires_grad', input_sentiment_embeddings.requires_grad)
            # print('output_sentiment_embedding_batch.requires_grad', output_sentiment_embedding_batch.requires_grad)
            l1 = (alpha) * loss_semantic_meaning(input_cls_embedding, output_cls_batch)
            l2 = (1 - alpha) * loss_sentiment(input_sentiment_embeddings, output_sentiment_embedding_batch)
            loss = l1 + l2
            print(f'Loss: {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    writer.close()

if __name__ == '__main__':
    bert_util = BertUtil()
    gpt2_util = GPT2Util()
    sentiment_analysis_util = SentimentAnalysisUtil()
    
    train_dataset = IMDBDataset(bert_util=bert_util, 
                            sentiment_analysis_util=sentiment_analysis_util, 
                            path=config.train_path)
    test_dataset = IMDBDataset(bert_util=bert_util, 
                            sentiment_analysis_util=sentiment_analysis_util, 
                            path=config.test_path)
    
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size)
    test_dl = DataLoader(test_dataset, batch_size=config.batch_size)

    model = Net()
    model.to(config.device)
    print('Trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters())

    if(config.train == True):
        train(model=model, 
            train_dl=train_dl, 
            test_dl=test_dl, 
            bert_util=bert_util, 
            sentiment_analysis_util=sentiment_analysis_util, 
            gpt2_util=gpt2_util, 
            optimizer=optimizer)
