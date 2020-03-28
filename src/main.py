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

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}")
        epoch_train_loss = 0
        for batch in tqdm(train_dl):
            input_cls_embedding = batch.cls_embedding
            input_word_embeddings = batch.word_embeddings
            input_sentiment_embeddings = batch.sentiment_embedding

            gpt2_input_embeds = model(input_cls_embedding, input_word_embeddings, input_sentiment_embeddings)

            for sentence in gpt2_util.batch_generate_sentence(gpt2_input_embeds):
                output_cls_embeddings = bert_util.generate_cls_embedding(sentence)
                output_word_embeddings = bert_util.generate_word_embeddings(sentence)
                output_sentiment_embedding = sentiment_analysis_util.get_sentiment_vector(sentence)

    writer.close()

if __name__ == '__main__':
    bert_util = BertUtil()
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

    optimizer = optim.Adam(model.parameters())

    if(config.train == True):
        train(model=model, 
            train_dl=train_dl, 
            test_dl=test_dl, 
            bert_util=bert_util, 
            sentiment_analysis_util=sentiment_analysis_util, 
            gpt2_util=gpt2_util, 
            optimizer=optimizer)
