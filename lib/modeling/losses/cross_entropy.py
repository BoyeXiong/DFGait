import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseLoss, gather_and_scale_wrapper 

class CrossEntropyLoss_2D(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(CrossEntropyLoss_2D, self).__init__(loss_term_weight)
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    @gather_and_scale_wrapper
    def forward(self, logits, labels):

        loss = self.loss(logits, labels)

        self.info.update({'loss': loss.detach().clone()})

        return loss, self.info