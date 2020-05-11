import torch
import torch.nn as nn

def loss_semantic(tokens_input, tokens_generated, sentence_bert):
    # Could also use cross entropy, WMD, cosine distance (recommended) or any other distance function
    cls_input = sentence_bert.generate_batch_sentence_embedding(tokens_input)
    cls_generated = sentence_bert.generate_batch_sentence_embedding(tokens_generated)
    cos = nn.CosineSimilarity(dim=1)
    return cos(cls_input, cls_generated)

def loss_sentiment(tokens_target, tokens_predicted, sent_analyser):
    # Cross entropy needs the target as the target class
    sentiment_target = sent_analyser.get_target_sentiment_vectors(tokens_target)
    sentiment_predicted = sent_analyser.get_batch_sentiment_vectors(tokens_predicted)
    
    cross_entropy = nn.CrossEntropyLoss()
    _, sentiment_target = sentiment_target.max(dim=1)
    
    # print(sentiment_predicted.shape, sentiment_target.shape)
    # quit()
    return cross_entropy(sentiment_predicted, sentiment_target)

def loss_mse_word_embeddings(input_word_embeddings, output_word_embeddings):
    mse = nn.MSELoss()
    return mse(input_word_embeddings, output_word_embeddings)