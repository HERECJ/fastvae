import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.cluster import KMeans

class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
    
    def encode_user(self, user_id):
        # return input
        raise NotImplementedError

    def encode_item(self, item_id):
        raise NotImplementedError
    
    def decode(self, input):
        # return input
        raise NotImplementedError

    def forward(self, *input):
        pass

    def klv_loss(self):
        pass

    def get_uv(self):
        # Obtain the latent matrix of users and items
        raise NotImplementedError

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
        super(VAE_CF, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.latent_dim = latent_dim

        self._User_Embedding_mu = nn.Embedding(self.num_user, self.latent_dim)
        self._User_Embedding_logvar = nn.Embedding(self.num_user, self.latent_dim)

        self._Item_Embedding_mu = nn.Embedding(self.num_item + 1, self.latent_dim)
        self._Item_Embedding_logvar = nn.Embedding(self.num_item + 1, self.latent_dim)
    
    def encode_user(self, user_id):
        return self.reparameterize(self._User_Embedding_mu(user_id), self._User_Embedding_logvar(user_id))
    
    def encode_item(self, item_id):
        return self.reparameterize(self._Item_Embedding_mu(item_id), self._Item_Embedding_logvar(item_id))
    
    # def forward(self, user_id, item_id):
        # user_vecs = self.encode_user(user_id)
        # item_vecs = self.encode_item(item_id)
        # return (user_vecs * item_vecs).sum(-1)
    def forward(self, user_id, pos_id, neg_ids):
        mask = (pos_id!=0).unsqueeze(dim=-1)
        user_vecs = self.encode_user(user_id).unsqueeze(dim=1)
        pos_items = self.encode_item(pos_id) * mask
        neg_items = self.encode_item(neg_ids)
        return (user_vecs * pos_items).sum(-1), (user_vecs * neg_items).sum(-1) 
    

    def klv_loss(self):
        """
        Compute the KLV for user embedding and user embedding respectively

        Return: kl_user, kl_item
        """
        return self.klv(mode=0), self.klv(mode=1)

    def klv(self, mode=0):
        """
        Compute the KL divergence loss according to Gaussian distributions

        mode: 0 for user embeddings
        mode: 1 for item embeddings
        """
        if mode < 1:
            return self._kl_gaussian(self._User_Embedding_mu.weight, self._User_Embedding_logvar.weight)
        else:
            return self._kl_gaussian(self._Item_Embedding_mu.weight[1:], self._Item_Embedding_logvar.weight[1:])
        
    def get_uv(self):
        # u_user = self.reparameterize(self._User_Embedding_mu.weight, self._User_Embedding_logvar.weight)
        # v_item = self.reparameterize(self._Item_Embedding_mu.weight, self._Item_Embedding_logvar.weight)
        u_user = self._User_Embedding_mu.weight
        v_item = self._Item_Embedding_mu.weight[1:]
        return u_user, v_item

# class Base_VAECF(VAE_CF):
#     def forward(self, user_id, user_history):
#         user_history = user_history.squeeze(dim=1)
#         user_vecs = self.encode_user(user_id)
#         all_items = self.reparameterize(self._Item_Embedding_mu.weight, self._Item_Embedding_logvar.weight)
#         kk = torch.matmul(user_vecs, all_items.T)
#         scores = torch.negative(F.log_softmax(kk, dim=-1))
#         return user_history * scores
        
class Base_VAECF(VAE_CF):
    def forward(self, user_id, user_history):
        user_history = user_history.squeeze(dim=1)
        user_vecs = self.encode_user(user_id)
        all_items = self.reparameterize(self._Item_Embedding_mu.weight, self._Item_Embedding_logvar.weight)
        
        # norm
        user_vecs_scale = ((user_vecs **2) + 1e-8).sum(-1, keepdim=True).sqrt()
        user_vecs = user_vecs / user_vecs_scale

        all_items_scale = ((all_items **2) + 1e-8).sum(-1, keepdim=True).sqrt()
        all_items = all_items / all_items_scale
        kk = torch.matmul(user_vecs, all_items.T)
        scores = torch.negative(F.log_softmax(kk, dim=-1))
        return user_history * scores

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





            
            


        


# class VAE_CF(BaseVAE):
#     def __init__(self, num_user, num_item, latent_dim, num_subspace, cluster_dim, **kwargs):
#         super(VAE_CF, self).__init__()
#         self.latent_dim = latent_dim
#         self.num_subspace = num_subspace
#         self.cluster_dim = cluster_dim
#         self.num_user = num_user
#         self.num_item = num_item

#         #encoder network
#         self._User_Embedding_mu = nn.Embedding(self.num_user, self.latent_dim)
#         self._User_Embedding_var = nn.Embedding(self.num_user, self.latent_dim)

#         # each item is represted by two quatized vector and an additional vector
#         # q_i = [V^1_i d^1_i; V^2_i d^2_i, \hat{q}_i]
#         # the dimension of each vector is cluster_dim, cluster_dim and latent_dim - 2 * cluster_dim
#         delta_dim = latent_dim - 2 * cluster_dim 
#         assert delta_dim >= 0

#         # the first sub space 
#         self._Cluster_1_Embedding_mu = nn.Embedding(self.num_subspace, self.cluster_dim)
#         self._Cluster_1_Embedding_var = nn.Embedding(self.num_subspace, self.cluster_dim)

#         self._Cluster_2_Embedding_mu = nn.Embedding(self.num_subspace, self.cluster_dim)
#         self._Cluster_2_Embedding_var = nn.Embedding(self.num_subspace, self.cluster_dim)

#         self._Item_Embedding_mu = nn.Embedding(self.num_item, delta_dim)
#         self._Item_Embedding_var = nn.Embedding(self.num_item, delta_dim)

#         #decoder network: Simple inner product
    
#     def encode_user(self, user_id):
#         return self._User_Embedding_mu(user_id), self._User_Embedding_var(user_id)
    
#     def encode_item(self, item_tuple):
#         [item_id, cluster_1_id, cluster_2_id] = item_tuple
#         item_mu = torch.cat([self._Cluster_1_Embedding_mu(cluster_1_id), self._Cluster_2_Embedding_mu(cluster_2_id), self._Item_Embedding_mu(item_id)], dim=-1)
#         item_var = torch.cat([self._Cluster_1_Embedding_var(cluster_1_id), self._Cluster_2_Embedding_var(cluster_2_id), self._Item_Embedding_var(item_id)], dim=-1)
#         return item_mu, item_var 

#     def reparameterize(self, mu, logvar):
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
    
#     def decode(user_emb, item_emb):
#         return (user_emb * item_emb).sum(-1)
    
#     def forward(self, user_id, item_tuple):
#         user_mu, user_var = self.encode_user(user_id)
#         item_mu, item_var = self.encode_item(item_tuple)
#         user_emb = self.reparameterize(user_mu, user_var)
#         item_emb = self.reparameterize(item_mu, item_var)
#         return self.decode(user_emb, item_emb)
    
#     def _KL_distance(self, mu, log_var):
#         # here we utilize the Gaussian distribution
#         # the prior distribution is the standard Gaussian
#         return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
#     def total_KL(self):
#         kl_user = self._KL_distance(self._User_Embedding_mu.weight, self._User_Embedding_var.weight)
#         kl_item = self._KL_distance(self._Item_Embedding_mu.weight, self._Item_Embedding_var.weight)
#         kl_item += self._KL_distance(self._Cluster_1_Embedding_mu.weight, self._Cluster_1_Embedding_var.weight)
#         kl_item += self._KL_distance(self._Cluster_2_Embedding_mu.weight, self._Cluster_2_Embedding_var.weight)
#         return kl_user + kl_item



# class QVAE_CF(BaseVAE):
#     """
#     The user embedding is quantized
#     While the item embedding is real values
#     """
#     def __init__(self, num_user, num_item, latent_dim, num_subspace, num_clusters, **kwargs):
#         super(QVAE_CF, self).__init__()
#         # 需不需要把这个改成params的形式
#         self.num_user = num_user
#         self.num_item = num_item
#         self.latent_dim = latent_dim
#         self.num_subspace = num_subspace
#         self.num_clusters = num_clusters
#         self._Item_Embedding_mu = nn.Embedding(self.num_item, self.latent_dim)
#         self._Item_Embedding_logvar = nn.Embedding(self.num_item, self.latent_dim)
        
#         # here we can add the gpu to the parameters
#         if (latent_dim % num_subspace) > 1:
#             raise ValueError('The space can not split %d sub spaces !'%num_subspace)
#         cluster_dim = math.floor(self.latent_dim / self.num_subspace)
#         self._User_Embedding = nn.ModuleDict()
#         for i in range(self.num_subspace):
#             self._User_Embedding[str(i)] = nn.Embedding(self.num_user, cluster_dim)

#     def encode_item(self, item_id):
#         return self.reparameterize(self._Item_Embedding_mu(item_id), self._Item_Embedding_logvar(item_id))
    
#     # def _encode_user_idx(self, user_emb, cluster_vecs):
#         # ruis = (user_emb * cluster_vecs).sum(-1)

#     def encode_user(self, user_id):
#         user_emb_dict = []
#         ruis_logits = []
#         for i in range(self.num_subspace):
#             user_emb = self._User_Embedding[str(i)](user_id) # N * dim
#             cluster_vec = self.clusters[i].cluster_centers_ # K * dim
#             ruis = torch.matmul(user_emb, cluster_vec)
#             ruis_logits.append(F.softmax(ruis)) # this is used for compute discrete KL divergence 
#             # Utilize the gumbel softmax trick to obtain the argmax 
#             ruis_idx_vec = F.gumbel_softmax(ruis, tau=1, hard=True, eps=1e-20) # N * K
#             # return the one-hot vector of K clusters
#             final_user_emb = torch.matmul(ruis_idx_vec, cluster_vec)
#             user_emb_dict.append(final_user_emb)
#         return torch.cat(user_emb_dict,dim=0), torch.cat(ruis_logits)
 
#     def get_item_emb(self):
#         item_id_array = torch.LongTensor(list(range(self.num_item)))
#         return  self.reparameterize(self._Item_Embedding_mu(item_id_array), self._Item_Embedding_logvar(item_id_array))
    
#     def compute_centroids(self):
#         self.item_vecs = self.get_item_emb()  # N * latent_dim
#         sub_spaces = self.item_vecs.chunk(self.num_subspace, dim=-1) # sub_spaces: tuple with the len of num_subspace
#         self.clusters = {}
#         for index, vecs in enumerate(sub_spaces):
#             # self.clusters[index] = perform_cluster_algorithms(vecs,self.num_clusters)
#             self.clusters[index] = KMeans(n_clusters=num_clusters,num_clusters=0).fit(vecs)
    
#     def reparameterize(self, mu, logvar):
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def forward(self, user_id, item_id):
#         # encode
#         self.compute_centroids()
#         user_vecs, user_logits_pos = self.encode_user(user_id)
#         item_vecs = self.encode_item(item_id)

#         #decode
#         # kl_loss_gaussian = self._kl_gaussian(self._Item_Embedding_mu, self._Item_Embedding_logvar)
#         kl_loss_multi = self._kl_user(user_logits_pos)
#         # (user_vecs * item_vecs).sum(-1)
#         return (user_vecs * item_vecs).sum(-1), kl_loss_gaussian, kl_loss_multi
    
#     def klv_loss(self):
#         return self._kl_gaussian(self._Item_Embedding_mu, self._Item_Embedding_logvar)
        
#     def _kl_user(self, user_logits_pos):
#         # Entropy of the logits
#         # user_logits_pos : num_subspace * ( N * num_centroids ) 
#         assert len(user_logits_pos) == self.num_subspace
#         num_user_batch, num_centroids = user_logits_pos[0].shape
#         assert num_centroids == self.num_clusters
#         p = 1.0 / num_centroids
#         pp = p.pow(self.num_subspace)
#         kl_loss = 0.0
#         for u in range(num_user_batch):
#             tmp = user_logits_pos[:, u, :] # num_subspace * K
#             res = 1.0
#             for idx in range(self.num_subspace):
#                 idx_vec = [1 for i in range(self.num_subspace)]
#                 idx_vec[idx] = num_centroids
#                 vec = tmp[idx].view(idx_vec)
#                 res = res * vec
#             kl_loss += self._kl_multinomial(res,pp)
#             # res : [num_centroids, num_centroids, ...]
#             # there are num_centroids^num_subspace items 
#         return kl_loss / num_user_batch
            
    
#     def _kl_multinomial(self, q_p, p, eps=1e-7):
#         h1 = q_p * torch.log(q_p + eps)
#         h2 = q_p * np.log(p + eps)
#         # kld_loss = torch.mean(torch.sum(h1 - h2, dim =(1,2)), dim=0)
#         return (h1 - h2).sum()
    
#     def _kl_gaussian(self, mu, log_var):
#         # here we utilize the Gaussian distribution
#         # the prior distribution is the standard Gaussian
#         return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)