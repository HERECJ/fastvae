import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.cluster import KMeans

class BaseVAE(nn.Module):
    def __init__(self, num_user, num_item, latent_dim, **kwargs):
        super(BaseVAE, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.latent_dim = latent_dim

        self._User_Embedding_mu = nn.Embedding(self.num_user, self.latent_dim)
        self._User_Embedding_logvar = nn.Embedding(self.num_user, self.latent_dim)

        self._Item_Embedding = nn.Embedding(self.num_item, self.latent_dim)

    def encode_user(self, user_id):
        return self.reparameterize(self._User_Embedding_mu(user_id), self._User_Embedding_logvar(user_id))

    def forward(self, user_id, user_history, neg_id, **kawargs):
        user_history = user_history.squeeze(dim=1)
        user_vecs = self.encode_user(user_id)
        all_items = self._Item_Embedding.weight
        
        # norm
        user_vecs_scale = ((user_vecs **2) + 1e-8).sum(-1, keepdim=True).sqrt()
        user_vecs = user_vecs / user_vecs_scale

        all_items_scale = ((all_items **2) + 1e-8).sum(-1, keepdim=True).sqrt()
        all_items = all_items / all_items_scale
        kk = torch.matmul(user_vecs, all_items.T)
        # scores = torch.negative(F.log_softmax(kk, dim=-1))
        return kk, kk

    def klv_loss(self):
        return self._kl_gaussian(self._User_Embedding_mu.weight, self._User_Embedding_logvar.weight)

    def get_uv(self):
        # Obtain the latent matrix of users and items
        u_user = self._User_Embedding_mu.weight
        v_item = self._Item_Embedding.weight
        return u_user, v_item

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _kl_gaussian(self, mu, log_var):
        # here we utilize the Gaussian distribution
        # the prior distribution is the standard Gaussian
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
class VAE_CF(BaseVAE):
    """
    The Model depending on real values of user and item vectors
    """
    def __init__(self, num_user, num_item, latent_dim, **kwargs):
        super(VAE_CF, self).__init__(num_user, num_item, latent_dim)
    
    def encode_user(self, user_id):
        return self.reparameterize(self._User_Embedding_mu(user_id), self._User_Embedding_logvar(user_id))
    
    def forward(self, user_id, user_history, neg_ids):
        user_vecs = self.encode_user(user_id)
        all_items = self._Item_Embedding.weight
        
        # norm
        user_vecs_scale = ((user_vecs **2) + 1e-8).sum(-1, keepdim=True).sqrt()
        user_vecs = user_vecs / user_vecs_scale

        all_items_scale = ((all_items **2) + 1e-8).sum(-1, keepdim=True).sqrt()
        all_items = all_items / all_items_scale
        kk = torch.matmul(user_vecs, all_items.T)
        
        neg_vecs = self._Item_Embedding(neg_ids)
        neg_vecs_scale = ((neg_vecs **2) + 1e-8).sum(-1, keepdim=True).sqrt()
        neg_vecs = neg_vecs / neg_vecs_scale

        return user_history * kk, (user_vecs.unsqueeze(1) * neg_vecs).sum(-1) 


class QVAE_CF(VAE_CF):
    """
    The user embedding is quantized
    """
    def __init__(self, num_user, num_item, latent_dim, num_partitions=2, num_centroids=32, **kwargs):
        # num_partitions is the number of splitted subspace
        # num_centroids is the number of centroids 
        super(QVAE_CF, self).__init__(num_user, num_item, latent_dim)
        self.num_partitions = num_partitions
        self.num_centroids = num_centroids

        if (latent_dim % self.num_partitions) > 1:
            raise ValueError('The space can not split %d sub spaces !'%self.num_partitions)
        self.cluster_dim = math.floor(self.latent_dim / self.num_partitions)
        self._User_Embedding = nn.Embedding(self.num_user, latent_dim)
        
        self._centroids_embedding = nn.ModuleDict()
        for i in range(self.num_partitions):
            self._centroids_embedding[str(i)] = nn.Embedding(self.num_centroids, self.cluster_dim)        
    
    def encode_user(self, user_ids):
        user_id = user_ids.squeeze(-1)
        batch_size = len(user_id)
        user_emb = self._User_Embedding(user_id)
        encode_emb = torch.tensor([])
        logits = torch.zeros((self.num_partitions, batch_size, self.num_centroids))
        cluster_idx_array = torch.LongTensor([i for i in range(self.num_centroids)])
        for i in range(self.num_partitions):
            start_idx = i * self.cluster_dim
            end_idx = (i + 1) * self.cluster_dim
            center_embs = self._centroids_embedding[str(i)](cluster_idx_array)
            distance = -self.compute_distance(user_emb[...,start_idx:end_idx], center_embs)
            logit = F.softmax(distance,dim=-1)
            idx_center = F.gumbel_softmax(distance, tau=1, hard=True)
            new_encode_emb = torch.cat((encode_emb, torch.matmul(idx_center, center_embs)), dim=1)
            encode_emb = new_encode_emb
            logits[i] = logit
        return encode_emb, logits
    
    # def forward(self, user_id, item_id):
        # import pdb; pdb.set_trace()
        # user_vecs, self.logits = self.encode_user(user_id)
        # item_vecs = self.encode_item(item_id)
        # return (user_vecs * item_vecs).sum(-1)

    def forward(self, user_id, pos_id, neg_ids):
        user_vec, self.logits = self.encode_user(user_id)
        user_vecs = user_vec.unsqueeze(1)
        pos_items = self.encode_item(pos_id)
        neg_items = self.encode_item(neg_ids)
        return (user_vecs * pos_items).sum(-1), (user_vecs * neg_items).sum(-1) 
    
    def klv_loss(self):
        return self._kl_user(self.logits),self.klv(mode=1)
    
    def _kl_user(self, user_logits_pos):
        assert len(user_logits_pos) == self.num_partitions
        num_user_batch, num_centroids = user_logits_pos[0].shape
        assert num_centroids == self.num_centroids
        p = 1.0 / num_centroids
        pp = p**self.num_partitions
        kl_loss = 0.0
        for u in range(num_user_batch):
            tmp = user_logits_pos[:, u, :] # num_partitions * K
            res = 1.0
            for idx in range(self.num_partitions):
                idx_vec = [1 for i in range(self.num_partitions)]
                idx_vec[idx] = num_centroids
                vec = tmp[idx].view(idx_vec)
                res = res * vec
            kl_loss += self._kl_multinomial(res,pp)
            # res : [num_centroids, num_centroids, ...]
            # there are num_centroids^num_partitions items 
        return kl_loss / num_user_batch



    def compute_distance(self, emb, centroids, mode=0):
        # compute the similarity between the embedding vector and the centroid vector
        # mode : {0 : L2-distance, 1 : consine similarity, 2 : inner product, 3: other}
        # emb : batch_size * cluster_dim
        # centroids : num_cluster * cluster_dim
        # output : batch_size * num_cluster
        if mode == 0:
            user_norm = (emb**2).sum(1).view(-1, 1)
            center_norm = (centroids**2).sum(1).view(1, -1)
            # return F.pairwise_distance(compare_emb, centroids, p=2.0)
            return user_norm + center_norm - 2.0 * torch.matmul(emb, centroids.T)
        elif mode == 1:
            user_norm = torch.norm(emb, dim=1).unsqueeze(1)
            center_norm = torch.norm(centroids, dim=1).unsqueeze(0)
            denom =  torch.matmul(user_norm, center_norm)
            return torch.matmul(emb, centroids.T) / denom
        elif mode == 2:
            return -torch.matmul(emb, centroids.T)
        else:
            raise NotImplementedError
    
    def _kl_multinomial(self, q_p, p, eps=1e-7):
        h1 = q_p * torch.log(q_p + eps)
        h2 = q_p * np.log(p + eps)
        # kld_loss = torch.mean(torch.sum(h1 - h2, dim =(1,2)), dim=0)
        return (h1 - h2).sum()
    
    def get_uv(self):
        # v_item = self.reparameterize(self._Item_Embedding_mu.weight, self._Item_Embedding_logvar.weight)
        v_item = self._Item_Embedding_mu
        user_idx_array = torch.LongTensor([i for i in range(self.num_user)])
        u_user, _ = self.encode_user(user_idx_array)
        return u_user, v_item