import torch
import torch.nn.functional as F
from torch import nn


class LogLoss(nn.Module):

    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm

    def forward(self, logits, labels):
        score = F.cross_entropy(logits,labels)
        if self.norm:
            priors = torch.bincount(labels,minlength=logits.size(1))/float(labels.shape[0])
            norm_factor = F.cross_entropy(torch.log(priors),labels)
        else:
            norm_factor = 1.0
        return score / norm_factor


class Brier(nn.Module):
    
    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm

    def forward(self, logits, labels):
        probs = F.softmax(logits,dim=1)
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1))
        score = F.mse_loss(probs,labels_onehot)

        if self.norm:
            priors = torch.bincount(labels,minlength=logits.size(1))/float(labels.shape[0])
            norm_factor = F.mse_loss(priors,labels_onehot)
        else:
            norm_factor = 1.0
        
        return score / norm_factor
