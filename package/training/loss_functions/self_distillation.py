import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjHead(nn.Module):
    def __init__(self):
        super(ProjHead, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head_Linear = nn.Sequential(
            nn.Linear(512*5*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.proj_head_Linear(out)
        return out

class self_distillation(nn.Module):
    def __init__(self, bs):
        super(self_distillation, self).__init__()
        self.bs = bs
        self.head = ProjHead().cuda()
        self.T = 0.5
        self.negatives_mask = self.negative_mask_correlated_samples(bs)
        # self.mask = self.mask_correlated_samples(bs)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def mask_correlated_samples(self, batch_size):
        K = 2 * batch_size
        mask = torch.ones((K, K), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def negative_mask_correlated_samples(self, batch_size):
        K = 2 * batch_size
        mask = torch.ones((K, K))
        mask = mask.fill_diagonal_(0)
        return mask

    def forward(self, ct_outs, mct_outs):
        feat_i = ct_outs[-1]
        feat_j = mct_outs[-1]
        K = 2 * self.bs
        feat_j = torch.flip(feat_j, dims=(3,))
        # feat_i = feat_i.view(self.bs, -1)
        # feat_j = feat_j.view(self.bs, -1)
        embd_i = self.head(feat_i)
        embd_j = self.head(feat_j)
        # embd_i = F.normalize(embd_i, dim=1)
        # embd_j = F.normalize(embd_j, dim=1)
        representations = torch.cat((embd_i, embd_j), dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.bs)
        sim_ji = torch.diag(similarity_matrix, -self.bs)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        # positives = torch.cat([sim_ij, sim_ji], dim=0).reshape(K, 1)
        # negatives = similarity_matrix[self.mask].reshape(K, -1)

        nominator = torch.exp(positives / self.T) 
        denominator = (self.negatives_mask.to(nominator.device)) * torch.exp(similarity_matrix / self.T)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / K
        return loss



class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out