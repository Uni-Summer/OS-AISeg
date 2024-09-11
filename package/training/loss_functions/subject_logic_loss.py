import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
import matplotlib.pyplot as plt

class SL_loss(nn.Module):
    def __init__(self, num_classes, batch_dice):
        super(SL_loss, self).__init__()
        self.c = num_classes
        self.act = softmax_helper
        self.annealing_step = 100
        self.dice = SDiceLoss()

    def KL(self, alpha, c):
        beta = torch.ones((1, c)).cuda()
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl
    
    def ce_loss(self, p, alpha, c, global_step, annealing_step):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = p
        # label = F.one_hot(p, num_classes=c)
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

        annealing_coef = min(1, global_step / annealing_step)

        alp = E * (1 - label) + 1
        B = annealing_coef * self.KL(alp, c)

        return (A + B)
    
    def UncBinaryDiceLoss(self, input, targets, unc):
        N = targets.size()[0]
        smooth = 1

        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        unc_flat = unc.view(N, -1)

        intersection = input_flat * targets_flat * unc_flat
        unc_input_flat = input_flat * unc_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (unc_input_flat.sum(1) + targets_flat.sum(1) + smooth)

        loss = 1 - N_dice_eff.sum() / N
        return loss

    def UncDiceLoss(self, input, target, w):
        total_loss = 0
        C = target.shape[1]

        for i in range(C):
            dice_loss = self.UncBinaryDiceLoss(input[:, i], target[:, i], w)
            total_loss += dice_loss

        return total_loss / C
    
    def get_soft_label(self, input_tensor, num_class):
        """
            convert a label tensor to soft label
            input_tensor: tensor with shape [N, C, H, W]
            output_tensor: shape [N, H, W, num_class]
        """
        tensor_list = []
        if input_tensor.ndim == 5:
            input_tensor = input_tensor.permute(0, 2, 3, 4, 1)
        else:
            input_tensor = input_tensor.permute(0, 2, 3, 1)
        for i in range(num_class):
            temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=-1)
        output_tensor = output_tensor.float()
        return output_tensor

    def forward(self, epoch, logits, target):
        evidence = F.softplus(logits)
        alpha = evidence + 1
        label = target.to(torch.int32)
        E = alpha -1
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        b = E / S
        unc = self.c / S
        one = torch.ones_like(unc)
        weight = one-unc
        weight = torch.squeeze(weight).cpu().detach().numpy()
    
        ## dice loss with uncertainty
        L_dice = self.UncDiceLoss(b, target, weight)
        ## or DiceLoss without uncertainty
        # L_dice = self.dice(evidence, target)
        ## or this way
        # evidence = self.act(evidence)
        # soft_l = self.get_soft_label(label, self.c)
        # L_dice = self.dice(evidence, soft_l)


        alpha = alpha.view(alpha.size(0), alpha.size(1), -1)
        alpha = alpha.transpose(1, 2) 
        alpha = alpha.contiguous().view(-1, alpha.size(2))
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(label, num_classes=self.c)
        label = label.view(-1, self.c)

        ## CE loss
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        L_ce = torch.mean(A)
        ## KL loss
        annealing_coef = min(1, epoch / self.annealing_step)
        alp = E * (1 - label) + 1
        B = annealing_coef * self.KL(alp, self.c)
        L_KL= torch.mean(B)

        loss = L_ce + L_KL + L_dice

        return prob, loss


class SDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(SDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets,weight_map=None):
        N = preds.size(0)
        C = preds.size(1)
        num_class = C
        if preds.ndim==5:
            preds = preds.permute(0, 2, 3, 4, 1)
        else:
            preds = preds.permute(0, 2, 3, 1)
        pred = preds.contiguous().view(-1, num_class)
        # pred = F.softmax(pred, dim=1)
        ground = targets.view(-1, num_class)
        n_voxels = ground.size(0)
        if weight_map is not None:
            weight_map = weight_map.view(-1)
            weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
            ref_vol = torch.sum(weight_map_nclass * ground, 0)
            intersect = torch.sum(weight_map_nclass * ground * pred, 0)
            seg_vol = torch.sum(weight_map_nclass * pred, 0)
        else:
            ref_vol = torch.sum(ground, 0)
            intersect = torch.sum(ground * pred, 0)
            seg_vol = torch.sum(pred, 0)
        dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
        # dice_loss = 1.0 - torch.mean(dice_score.data[1:dice_score.shape[0]])
        k = -torch.log(dice_score)
        # 1. mean-loss
        dice_mean_score = torch.mean(-torch.log(dice_score))
        # 2. sum-loss
        # dice_score1 = torch.sum(dice_score,0)
        # dice_mean_score1 = -torch.log(torch.sum(dice_score,0))
        # dice_mean_score = torch.mean(-torch.log(dice_score.data[1:dice_score.shape[0]]))
        return dice_mean_score
        



