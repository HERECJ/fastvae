import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.cluster import KMeans

class BaseVAE(nn.Module):
    def __init__(self, num_item, dims, active='relu', dropout=0.5):
        """
        dims is a list for latent dims
        """
        super(BaseVAE, self).__init__()
        self.num_item = num_item
        if len(dims) < 2:
            self.dims = [num_item] + [dims[-1] *2]
        else:
            self.dims = [num_item] + dims[:-1] + dims[-1] *2 
        # init encode
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.dropout = nn.Dropout(dropout)

        # init decode
        self._Item_Embeddings = nn.Embedding(self.num_item, self.dims[-1]//2)

        if active == 'relu':
            self.act = F.relu
        elif active == 'tanh':
            self.act == F.tanh
        elif active == 'sigmoid':
            self.act == F.sigmoid
        else:
            raise ValueError('Not supported active function')

    
    def encode(self, user_his):
        # user_his is the vector of user histories, 1 * N
        h = user_his
        h = F.normalize(user_his)
        h = self.dropout(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = self.act(h)
            else:
                latent_dim = self.dims[-1]//2
                mu = h[:, :latent_dim]
                logvar = h[:, latent_dim:]
        return mu, logvar
    
    def decode(self, user_emb, items):
        item_embs = self._Item_Embeddings(items)
        # item_embs = F.normalize(item_embs)
        return torch.matmul(user_emb, item_embs.T)
    

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    
    def forward(self, user_his, pos_items, neg_items, device='cuda'):
        mu, logvar = self.encode(user_his)
        z = self.reparameterize(mu, logvar)

        items = torch.LongTensor(range(self.num_item))
        if device=='cuda':
            items = items.cuda()
        return  None, self.decode(z, items), mu, logvar
    
    def kl_loss(self, mu, log_var, anneal=1.0):
        return -anneal * 0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)
    
    def loss_function(self, user_his, part_rats, prob_neg=None, pos_rats=None, prob_pos=None, reduction=False):
        if reduction is True:
            return -torch.sum((F.log_softmax(part_rats, dim=1) * user_his), dim=-1).mean()
        else:
            return -torch.sum((F.log_softmax(part_rats, dim=1) * user_his), dim=-1).sum()
 

    def _get_user_emb(self, user_his):
        user_emb, _ = self.encode(user_his)
        return user_emb

    def _get_item_emb(self):
        return self._Item_Embeddings.weight
    
        
class VAE_Sampler(BaseVAE):
    def __init__(self, num_item, dims, active='relu', dropout=0.5):
        super(VAE_Sampler, self).__init__(num_item, dims, active='relu', dropout=0.5)
        self._Item_Embeddings = nn.Embedding(self.num_item + 1, self.dims[-1]//2, padding_idx=0)  # embedding 0 for padding
    
    def decode(self, user_emb, items):
        item_embs = self._Item_Embeddings(items)
        return (user_emb.unsqueeze(1) * item_embs).sum(-1)


    def forward(self, user_his, pos_items, neg_items):
        mu, logvar = self.encode(user_his)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, pos_items), self.decode(z, neg_items), mu, logvar
    
    def loss_function(self, user_his, part_rats, log_prob_neg=None, pos_rats=None, log_prob_pos=None, reduction=False):
        new_pos = pos_rats - log_prob_pos.detach()
        new_neg = part_rats - log_prob_neg.detach()
        parts_sum_exp = torch.sum(torch.exp(new_neg), dim=-1).unsqueeze(-1)
        new_pos[pos_rats==0] = -float("Inf")
        final = torch.log( torch.exp(new_pos) + parts_sum_exp)
        if reduction is True:
            return torch.mean((- new_pos + final)[pos_rats!=0], dim=-1 )
        else:
            return torch.sum((- new_pos + final)[pos_rats!=0], dim=-1 )
    
    def _get_item_emb(self):
        return self._Item_Embeddings.weight[1:]