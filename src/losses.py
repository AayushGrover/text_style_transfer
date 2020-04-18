import torch
import torch.nn as nn

pdist = nn.PairwiseDistance(p=2)
centropy = nn.CrossEntropyLoss()

def loss_semantic_meaning(cls_input, cls_generated):
    # could also use cross entropy, WMD, cosine distance (recommended) or any other distance function
    return pdist(cls_input,cls_generated)

def loss_sentiment(sentiment_target, sentiment_predicted):
    # Cross entropy needs the target as the target class
    _,sentiment_target = sentiment_target.max(dim=1)
    return centropy(sentiment_predicted, sentiment_target)
