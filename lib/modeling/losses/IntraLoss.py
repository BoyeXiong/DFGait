import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseLoss, gather_and_scale_wrapper

class IntraLoss(BaseLoss):

    def __init__(self, loss_term_weight=1.0):
        super(IntraLoss, self).__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1).float()
        input2 = input2.view(batch_size, -1).float()

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        self.info.update({'loss': diff_loss.detach().clone()})

        return diff_loss, self.info

