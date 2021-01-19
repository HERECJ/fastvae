from sklearn.cluster import KMeans
import scipy as scp
import numpy as np
import pdb
import torch

class sampler(object):
    def __init__(self, user_embs, item_embs, num_subspace, cluster_dim, num_cluster, res_dim=0):
        # item_embs : N * K
        code_dict = {}
        item_indices = {}
        self.center_scores = {}
        self.num_items, latent_dim = item_embs.shape
        self.combine_cluster_idx = torch.zeros((self.num_items))
        self.num_cluster = num_cluster
        self.num_subspace = num_subspace
        for i in range(self.num_subspace):
            start_idx = i * cluster_dim
            end_idx = (i+1) * cluster_dim
            cluster_kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(item_embs[:,start_idx:end_idx].detach().numpy())
            centers = cluster_kmeans.cluster_centers_
            codes = cluster_kmeans.labels_
            
            # code_mat = scp.sparse.csr_matrix((np.ones_like(codes), (np.arange(self.num_items), codes)), shape=(self.num_items, self.num_cluster))
            # num_items_in_cluster = code_mat.sum(axis=0).A #calculate the number of items in each cluster
            
            # code_mat = code_mat.tocsc()
            # items_in_cluster = [np.array([code_mat.indices[j] for j in range(code_mat.indptr[c], code_mat.indptr[c+1])]) for c in range(self.num_cluster)]

            # code_dict[i] = codes
            # item_indices[i] = items_in_cluster
            self.center_scores[i] = torch.matmul(user_embs[:,start_idx:end_idx] , torch.tensor(centers).T)
            self.combine_cluster_idx = self.combine_cluster_idx * self.num_cluster + codes
            # the first is K, second K'

            # Use a len of K^S vector to denote the index of clusters 
        
        # the inner product of residual vectors
        self.delta_ruis = torch.matmul(user_embs[:,end_idx:], item_embs[:,end_idx:].T)

        # code_combine_mat = scp.sparse.csr_matrix((np.ones_like(codes), (np.arange(self.num_items), self.combine_cluster_idx)), shape=(self.num_items, self.num_cluster**self.num_subspace))

        # code_combine_trans = code_combine_mat.tocsc()
        # items_in_combine_cluster = [np.array([code_combine_trans.indices[j] for j in range(code_combine_trans.indptr[c], code_combine_trans.indptr[c+1])]) for c in range(self.num_cluster**self.num_subspace)]
        # for two subspace
    
    def preprocess(self, user_id):
        self.user_id = user_id
        delta_ruis_u = torch.exp(self.delta_ruis[user_id]).detach().numpy()
        combine_mat = scp.sparse.csr_matrix((delta_ruis_u, (np.arange(self.num_items), self.combine_cluster_idx)), shape=(self.num_items, self.num_cluster**self.num_subspace))

        code_mat = combine_mat.tocsc()
        self.items_in_combine_cluster = [np.array([code_mat.indices[j] for j in range(code_mat.indptr[c], code_mat.indptr[c+1])]) for c in range(self.num_cluster**self.num_subspace)]
        # w_kk : \sum_{j\in K K'} exp(rui)
        self.w_kk = np.squeeze(combine_mat.sum(axis=0).A)


        shape_list = [self.num_cluster for x in range(self.num_subspace)]
        kk_mtx = self.w_kk.reshape(shape_list)
        self.sample_prob = {}
        self.sample_prob[self.num_subspace-1] = torch.tensor(kk_mtx)
        for i in range(self.num_subspace-2, -1, -1):
            r_centers = torch.exp(self.center_scores[i][user_id]).unsqueeze(-1).detach().numpy()
            kk_mtx = np.matmul(kk_mtx, r_centers)
            self.sample_prob[i] = torch.tensor(kk_mtx)

    
    def __sampler__(self, pos_id=0):
        idx = []
        for i in range(self.num_subspace):
            sample_probs = self.sample_prob[i]
            if len(idx) > 0:
                for history_cluster in idx:
                    sample_probs = sample_probs[history_cluster]
            extra = sample_probs.squeeze()
            idx_cluster = self.sample_from_gumbel_noise(self.center_scores[i][self.user_id], torch.log(extra))
            idx.append(idx_cluster)
        import pdb; pdb.set_trace()
        # sample from the final items
        index_combine_cluster = 0
        for ii in idx:
            index_combine_cluster = index_combine_cluster * self.num_cluster + ii
        import pdb; pdb.set_trace()
        items_index = self.items_in_combine_cluster[index_combine_cluster]
        rui_items = self.delta_ruis[self.user_id][items_index]
        item_sample_index = self.sample_from_gumbel_noise(rui_items)
        return items_index[item_sample_index]

    
    def sample_gumbel_noise(self, input,eps=1e-7):
        u = torch.rand(input.shape)
        return -torch.log(eps - torch.log(u + eps))
    
    def sample_from_gumbel_noise(self, scores, extra=None):
        if extra is None:
            extra = torch.zeros_like(scores)
        return torch.argmax(scores + extra + self.sample_gumbel_noise(scores))