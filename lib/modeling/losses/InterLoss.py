import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper

class InterLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(InterLoss, self).__init__(loss_term_weight)

    def loss_kld(self, y_s, y_t):
        p_s = F.log_softmax(y_s, dim=1)
        p_t = F.softmax(y_t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean')
        return loss

    @gather_and_scale_wrapper
    def forward(self, sil_l, sil_g, ske_l, ske_g, labels):
        bsz = sil_l.shape[0]
        sil_l = sil_l.view(bsz, -1)
        sil_g = sil_g.view(bsz, -1)
        ske_l = ske_l.view(bsz, -1)
        ske_g = ske_g.view(bsz, -1)
        l = (self.loss_kld(sil_l, ske_l) + self.loss_kld(ske_l, sil_l)) / 2
        g = (self.loss_kld(sil_g, ske_g) + self.loss_kld(ske_g, sil_g)) / 2
        loss = g / l
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info

    