import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.utilities.nd_softmax import softmax_helper

class Barlow_twins_loss(nn.Module):
    def __init__(self, batch_size, lambd=0.005):
        super(Barlow_twins_loss, self).__init__()
        self.batch_size = batch_size
        self.lambd = lambd
        self.apply_nonlin = softmax_helper
    
    def _dice(self, score, target):
        smooth = 1e-6
        dim_len = len(score.size())
        if dim_len == 5:
            dim=(2,3,4)
        elif dim_len == 4:
            dim=(2,3)
        intersect = torch.sum(score * target,dim=dim)
        y_sum = torch.sum(target * target,dim=dim)
        z_sum = torch.sum(score * score,dim=dim)
        dice_sim = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_sim

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

    def forward(self, input, target):
        mm_evidence = F.softplus(target)
        mm_alpha = mm_evidence + 1
        mm_S = torch.sum(mm_alpha, dim=1, keepdim=True)
        confidence = torch.exp(-(self.c / mm_S))

        input = self.apply_nonlin(input+1)
        target = self.apply_nonlin((target+1) * confidence)
        z1 = torch.cat([input, target], dim=0)
        z2 = torch.cat([target, input], dim=0)
        assert z1.size()[2:] == z2.size()[2:], 'predict & target shape do not match'
        D = self._dice(z1.unsqueeze(1), z2.unsqueeze(0))
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool).to(z1.device)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        D = D * mask
        on_diag = torch.diagonal(D).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(D).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss
        