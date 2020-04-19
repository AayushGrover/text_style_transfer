import torch
import torch.nn as nn

def loss_semantic_meaning(cls_input, cls_generated):
    # Could also use cross entropy, WMD, cosine distance (recommended) or any other distance function
    cos = nn.CosineSimilarity()
    return cos(cls_input, cls_generated)

def loss_sentiment(sentiment_target, sentiment_predicted):
    cross_entropy = nn.CrossEntropyLoss()
    # Cross entropy needs the target as the target class
    _, sentiment_target = sentiment_target.max(dim=1)
    return cross_entropy(sentiment_predicted, sentiment_target)
