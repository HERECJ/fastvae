import numpy as np
import scipy as sp
import scipy.sparse
import scipy.special
from sklearn.cluster import KMeans
import bisect
import random
import torch
import torch.nn.functional as F
import math

class SamplerUserModel:
    """
    For each user, sample negative items
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        self.mat = mat.tocsr()
        self.num_users, self.num_items = mat.shape
        self.num_neg = num_neg

    def preprocess(self, user_id):
        # self.exist = set(self.mat.indices[j] for j in range(self.mat.indptr[user_id], self.mat.indptr[user_id + 1]))
        pass

    def __sampler__(self, user_id):
        def sample():
            return list(torch.utils.data.RandomSampler(range(self.num_items), num_samples=self.num_neg, replacement=True)), [np.log(1.0/self.num_items) for _ in range(self.num_neg)]
        return sample

    def negative_sampler(self, start_id, end_id):
        def sample_negative(user_id):
            sample = self.__sampler__(user_id)
            k, p = sample()
            return k, p

        def generate_tuples():
            for i in np.random.permutation(range(start_id, end_id)):
                self.preprocess(i)
                neg_item, prob = sample_negative(i)
                user_his = torch.tensor(self.mat[i].toarray()[0].tolist())
                pos_idx = self.mat[i].nonzero()[1]
                pos_prob = self.compute_item_p(i, pos_idx)
                user_prob = self.mat[i].toarray()[0]
                user_prob[pos_idx] = pos_prob
                yield i, user_his, torch.tensor(user_prob), torch.LongTensor(neg_item), torch.tensor(prob)
        return generate_tuples
    
    def compute_item_p(self, user_id, item_list):
        return [np.log(1.0/self.num_items) for _ in item_list]


class PopularSamplerModel(SamplerUserModel):
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster, mode=0):
        super(PopularSamplerModel, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        pop_count = np.squeeze(mat.sum(axis=0).A)
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_prob = pop_count / np.sum(pop_count)
        self.pop_cum_prob = self.pop_prob.cumsum()

    def __sampler__(self, user_id):
        def sample():
            neg_items = []
            probs = []
            seeds = torch.rand(self.num_neg)
            for s in seeds:
                k = bisect.bisect(self.pop_cum_prob, s)
                p = np.log(self.pop_prob[k])
                neg_items.append(k)
                probs.append(p)
            return neg_items, probs
        return sample
    
    def compute_item_p(self, user_id, item_list):
        return [np.log(self.pop_prob[k]) for k in item_list]

class ExactSamplerModel(SamplerUserModel):
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        super(ExactSamplerModel, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        self.user_embs = user_embs.cpu().detach().numpy()
        self.item_embs = item_embs.cpu().detach().numpy()

    def preprocess(self, user_id):
        super(ExactSamplerModel, self).preprocess(user_id)
        pred = self.user_embs[user_id] @ self.item_embs.T
        idx = np.argpartition(pred, -5)[-5:]
        pred[idx] = -np.inf
        self.score = sp.special.softmax(pred)
        self.score_cum = self.score.cumsum()
        self.score_cum[-1] = 1.0
        # print(user_id)
        
    def __sampler__(self, user_id):
        def sample():
            neg_items = []
            probs = []
            seeds = torch.rand(self.num_neg)
            for s in seeds:
                k = bisect.bisect(self.score_cum,s)
                p = np.log(self.score[k])
                neg_items.append(k)
                probs.append(p)
            return neg_items, probs
        return sample

    def compute_item_p(self, user_id, item_list):
        return [np.log(self.score[k]) for k in item_list]


class SoftmaxApprSampler(SamplerUserModel):
    """
    PQ methods, each item vector is splitted into three parts
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        super(SoftmaxApprSampler, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        self.num_cluster = num_cluster
        
        _, latent_dim = user_embs.shape
        
        cluster_dim = latent_dim // 2
        

        user_embs = user_embs.cpu().data.detach().numpy()
        item_embs = item_embs.cpu().data.detach().numpy()
        self.user_embs = user_embs


        cluster_kmeans_0 = KMeans(n_clusters=self.num_cluster, random_state=0).fit(item_embs[:,:cluster_dim])
        centers_0 = cluster_kmeans_0.cluster_centers_
        codes_0 = cluster_kmeans_0.labels_
        self.center_scores_0 = np.matmul(user_embs[:,:cluster_dim] , centers_0.T)
        

        cluster_kmeans_1 = KMeans(n_clusters=self.num_cluster, random_state=0).fit(item_embs[:,cluster_dim:])
        centers_1 =cluster_kmeans_1.cluster_centers_
        codes_1 = cluster_kmeans_1.labels_
        self.center_scores_1 = np.matmul(user_embs[:,cluster_dim:] , centers_1.T)


        self.union_idx = [codes_0[i] * num_cluster + codes_1[i] for i in range(self.num_items)]

        self.combine_cluster = sp.sparse.csc_matrix((np.ones_like(self.union_idx), (np.arange(self.num_items), self.union_idx)), shape=(self.num_items, self.num_cluster**2))
        combine_sum = np.sum(self.combine_cluster, axis=0).A
        self.idx_nonzero = combine_sum > 0


        cluster_emb_0 = centers_0[codes_0]
        cluster_emb_1 = centers_1[codes_1]
        cluster_emb = np.concatenate((cluster_emb_0, cluster_emb_1), axis=1)
        self.item_emb_res = item_embs - cluster_emb

    def preprocess(self, user_id):
        
        delta_rui = np.matmul(self.user_embs[user_id], self.item_emb_res.T)
        combine_tmp = sp.sparse.csr_matrix((np.exp(delta_rui), (np.arange(self.num_items), np.arange(self.num_items))), shape=(self.num_items, self.num_items))
        combine_mat = combine_tmp * self.combine_cluster

        # combine_max = np.max(combine_mat, axis=0).A
        # combine_norm = 


        # w_kk : \sum_{i\in K K'} exp(rui)
        w_kk = combine_mat.sum(axis=0).A
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        self.kk_mtx = w_kk.reshape((self.num_cluster, self.num_cluster))
        
        r_centers_1 = self.center_scores_1[user_id]
        phi_k_tmp = self.kk_mtx  +  r_centers_1
        self.p_table_1 = (sp.special.softmax(phi_k_tmp, 1)).cumsum(axis=1)

        phi_k = np.sum(np.exp(phi_k_tmp), axis=1)

        r_centers_0 = self.center_scores_0[user_id]
        self.p_table_0 = (sp.special.softmax(r_centers_0 + np.log(phi_k))).cumsum()
        
        self.partition = np.sum(np.exp(r_centers_0 + np.log(phi_k)))
    
    def compute_item_p(self, user_id, item_list):
        clusters_idx = self.combine_cluster[item_list, :].nonzero()[1] # find the combine cluster idx
        k_0 = clusters_idx // self.num_cluster
        k_1 = clusters_idx % self.num_cluster
        
        p_0 = np.array(self.comput_p(k_0, self.p_table_0))
        p_1 = np.squeeze(np.array([self.comput_p(np.array([idx_1]), self.p_table_1[idx_0]) for idx_1, idx_0 in zip(k_1, k_0)]))

        frac = np.matmul(self.user_embs[user_id], self.item_emb_res[item_list].T)
        deno = np.array([self.kk_mtx[idx_0, idx_1] for idx_0, idx_1 in zip(k_0, k_1)])
        # p_r = np.exp(frac - deno)
        return np.log(p_0) + np.log(p_1) + frac - deno
        
    

    def comput_p(self, sampled_cluster, p_table):
        p_list = p_table[sampled_cluster]
        p_list_former = p_table[sampled_cluster - 1]
        return [ x - y  if x> y else x for x,y in zip(p_list, p_list_former)]

    def __sampler__(self, user_id):
        def sample():
            seeds_values = np.random.rand(self.num_neg)
            sampled_cluster_0 = np.array(list(map(lambda x : bisect.bisect(self.p_table_0, x) , seeds_values)))
            p_0 = np.array(self.comput_p(sampled_cluster_0, self.p_table_0))


            seeds_values = np.random.rand(self.num_neg)
            sampled_cluster_1 = list(map(lambda idx, x: bisect.bisect(self.p_table_1[idx], x) , sampled_cluster_0, seeds_values))
            
            p_1 = np.squeeze(np.array([self.comput_p(np.array([idx_1]), self.p_table_1[idx_0]) for idx_1, idx_0  in zip(sampled_cluster_1, sampled_cluster_0)]))
            

            idx_final_cluster = [idx_0 * self.num_cluster + idx_1 for idx_0, idx_1 in zip(sampled_cluster_0, sampled_cluster_1)]

            idx_items_lst = [[ self.combine_cluster.indices[j] for j in range(self.combine_cluster.indptr[c], self.combine_cluster.indptr[c+1])] for c in idx_final_cluster]
            
            final_items = []
            final_probs = []
            for items in idx_items_lst:
                rui_items =  np.matmul(self.user_embs[user_id], self.item_emb_res[items].T)
                # to compute on the fly without delta
                item_sample_idx, p = self.sample_final_items(rui_items)
                final_items.append(items[item_sample_idx])
                final_probs.append(p)
            
            final_probs = np.log(p_0) + np.log(p_1) +  np.log(np.array(final_probs))
            return final_items, final_probs
        return sample
    
    def negative_sampler(self, start_id, end_id):
        def sample_negative(user_id):
            sample = self.__sampler__(user_id)
            k, p = sample()
            return k, p

        def generate_tuples():
            for i in np.random.permutation(range(start_id, end_id)):
                self.preprocess(i)
                neg_item, prob = sample_negative(i)
                # uid_emb = self.user_embs[i]
                user_his = torch.tensor(self.mat[i].toarray()[0].tolist())
                pos_idx = self.mat[i].nonzero()[1]
                pos_prob = self.compute_item_p(i, pos_idx)
                user_prob = self.mat[i].toarray()[0]
                user_prob[pos_idx] = pos_prob
                yield i, user_his, torch.tensor(user_prob), torch.LongTensor(neg_item), torch.tensor(prob)
        return generate_tuples

    def sample_final_items(self, scores, eps=1e-8, mode=0):
        pred = sp.special.softmax(scores)
        if mode == 0:
            # Gumbel noise
            us = np.random.rand(len(scores))
            tmp = scores - np.log(- np.log(us + eps) + eps)
            k = np.argmax(tmp)
            return k, pred[k] 
        elif mode == 1:
            score_cum = pred.cumsum()
            k = bisect.bisect(score_cum, np.random.rand()) 
            return k, pred[k]

class SoftmaxApprSamplerUniform(SoftmaxApprSampler):
    """
    Uniform sampling for the final items
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        super(SoftmaxApprSamplerUniform, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        
        w_kk = np.float32(self.combine_cluster.sum(axis=0).A)
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        self.kk_mtx = w_kk.reshape((self.num_cluster, self.num_cluster))
        
    
    def preprocess(self, user_id):
        r_centers_1 = self.center_scores_1[user_id]
        phi_k_tmp = self.kk_mtx  +  r_centers_1
        self.p_table_1 = (sp.special.softmax(phi_k_tmp, 1)).cumsum(axis=1)

        phi_k = np.sum(np.exp(phi_k_tmp), axis=1)

        r_centers_0 = self.center_scores_0[user_id]
        self.p_table_0 = (sp.special.softmax(r_centers_0 + np.log(phi_k))).cumsum()
        
        self.partition = np.sum(np.exp(r_centers_0 + np.log(phi_k)))


    def __sampler__(self, user_id):
        def sample():
            seeds_values = np.random.rand(self.num_neg)
            sampled_cluster_0 = np.array(list(map(lambda x : bisect.bisect(self.p_table_0, x) , seeds_values)))
            p_0 = np.array(self.comput_p(sampled_cluster_0, self.p_table_0))


            seeds_values = np.random.rand(self.num_neg)
            sampled_cluster_1 = list(map(lambda idx, x: bisect.bisect(self.p_table_1[idx], x) , sampled_cluster_0, seeds_values))
            
            p_1 = np.squeeze(np.array([self.comput_p(np.array([idx_1]), self.p_table_1[idx_0]) for idx_1, idx_0  in zip(sampled_cluster_1, sampled_cluster_0)]))
            

            idx_final_cluster = [idx_0 * self.num_cluster + idx_1 for idx_0, idx_1 in zip(sampled_cluster_0, sampled_cluster_1)]
            
            idx_items_lst = [[ self.combine_cluster.indices[j] for j in range(self.combine_cluster.indptr[c], self.combine_cluster.indptr[c+1])] for c in idx_final_cluster]

 
            final_items = [np.random.choice(items) for items in idx_items_lst] 
            final_probs = [ 1.0 / len(items) for items in idx_items_lst]
            final_probs = np.log(p_0) + np.log(p_1) +  np.log(np.array(final_probs))
            return final_items, final_probs
        return sample
    
    def compute_item_p(self, user_id, item_list):
        clusters_idx = self.combine_cluster[item_list, :].nonzero()[1] # find the combine cluster idx
        k_0 = clusters_idx // self.num_cluster
        k_1 = clusters_idx % self.num_cluster
        
        p_0 = np.array(self.comput_p(k_0, self.p_table_0))
        p_1 = np.squeeze(np.array([self.comput_p(np.array([idx_1]), self.p_table_1[idx_0]) for idx_1, idx_0 in zip(k_1, k_0)]))
        
        # p_r =


        deno = np.array([self.kk_mtx[idx_0, idx_1] for idx_0, idx_1 in zip(k_0, k_1)])
        # p_r = np.exp( - deno)
        return np.log(p_0) + np.log(p_1) - deno

class SoftmaxApprSamplerPop(SoftmaxApprSampler):
    """
    Popularity sampling for the final items
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster, mode=0):
        super(SoftmaxApprSamplerPop, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        pop_count = np.squeeze(mat.sum(axis=0).A)
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        
        self.pop_count = pop_count
        
        
        combine_tmp = sp.sparse.csr_matrix((self.pop_count, (np.arange(self.num_items), np.arange(self.num_items))), shape=(self.num_items, self.num_items))
        combine_mat = combine_tmp * self.combine_cluster

        w_kk = combine_mat.sum(axis=0).A

        w_k_tmp = sp.sparse.csr_matrix(( 1.0/(np.squeeze(w_kk) + np.finfo(float).eps), (np.arange(self.num_cluster**2), np.arange(self.num_cluster**2))), shape=(self.num_cluster**2, self.num_cluster**2))
        
        self.pop_probs_mat = (combine_mat * w_k_tmp).tocsc()
        
        self.pop_cum_mat = self.pop_probs_mat.copy()
        for c in range(self.num_cluster**2):
            item_prob = [ self.pop_probs_mat.data[j] for j in range(self.pop_probs_mat.indptr[c], self.pop_probs_mat.indptr[c+1])]
            item_cum_prob = np.cumsum(np.array(item_prob))
            for item_i, j in enumerate(range(self.pop_probs_mat.indptr[c], self.pop_probs_mat.indptr[c+1])):
                self.pop_cum_mat.data[j] = item_cum_prob[item_i]        
        
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        self.kk_mtx = w_kk.reshape((self.num_cluster, self.num_cluster))



    def preprocess(self, user_id):
        r_centers_1 = self.center_scores_1[user_id]
        phi_k_tmp = self.kk_mtx  +  r_centers_1
        self.p_table_1 = (sp.special.softmax(phi_k_tmp, 1)).cumsum(axis=1)

        phi_k = np.sum(np.exp(phi_k_tmp), axis=1)

        r_centers_0 = self.center_scores_0[user_id]
        self.p_table_0 = (sp.special.softmax(r_centers_0 + np.log(phi_k))).cumsum()
        
        self.partition = np.sum(np.exp(r_centers_0 + np.log(phi_k)))
    
    def __sampler__(self, user_id):
        def sample():
            seeds_values = np.random.rand(self.num_neg)
            sampled_cluster_0 = np.array(list(map(lambda x : bisect.bisect(self.p_table_0, x) , seeds_values)))
            p_0 = np.array(self.comput_p(sampled_cluster_0, self.p_table_0))


            seeds_values = np.random.rand(self.num_neg)
            sampled_cluster_1 = list(map(lambda idx, x: bisect.bisect(self.p_table_1[idx], x) , sampled_cluster_0, seeds_values))
            
            p_1 = np.squeeze(np.array([self.comput_p(np.array([idx_1]), self.p_table_1[idx_0]) for idx_1, idx_0  in zip(sampled_cluster_1, sampled_cluster_0)]))
            

            idx_final_cluster = [idx_0 * self.num_cluster + idx_1 for idx_0, idx_1 in zip(sampled_cluster_0, sampled_cluster_1)]
            
            # idx_items_lst = [[ self.combine_cluster.indices[j] for j in range(self.combine_cluster.indptr[c], self.combine_cluster.indptr[c+1])] for c in idx_final_cluster]

            final_items, final_probs = [], []
            for c in idx_final_cluster:
                items, cum_probs = zip(*[ (self.pop_cum_mat.indices[j], self.pop_cum_mat.data[j]) for j in range(self.pop_probs_mat.indptr[c], self.pop_probs_mat.indptr[c+1])])
                k = bisect.bisect(cum_probs, np.random.rand())
                sampled_item = items[k]

                p_r = self.pop_probs_mat[sampled_item, c]
                final_items.append(sampled_item)
                final_probs.append(p_r)

            final_probs = np.log(p_0) + np.log(p_1) +  np.log(np.array(final_probs))
            return  final_items, final_probs
        return sample
    
    def compute_item_p(self, user_id, item_list):
        clusters_idx = self.combine_cluster[item_list, :].nonzero()[1] # find the combine cluster idx
        k_0 = clusters_idx // self.num_cluster
        k_1 = clusters_idx % self.num_cluster
        
        p_0 = np.array(self.comput_p(k_0, self.p_table_0))
        p_1 = np.squeeze(np.array([self.comput_p(np.array([idx_1]), self.p_table_1[idx_0]) for idx_1, idx_0 in zip(k_1, k_0)]))
        
        # p_r =
        p_r = np.squeeze(np.array([self.pop_probs_mat[idx,:].data for idx in item_list ]))
        # p_r = np.array(self.pop_probs_mat[item_list,:].data)
        return np.log(p_0) + np.log(p_1) + np.log(p_r)

class UniformSoftmaxSampler(SoftmaxApprSamplerUniform):
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        super(UniformSoftmaxSampler, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
    
    def preprocess(self, user_id):
        r_centers_1 = self.center_scores_1[user_id]
        phi_k_tmp = self.kk_mtx  +  r_centers_1
        self.p_table_1 = (sp.special.softmax(phi_k_tmp, 1)).cumsum(axis=1)

        phi_k = np.sum(np.exp(phi_k_tmp), axis=1)

        r_centers_0 = self.center_scores_0[user_id]
        self.p_table_0 = (sp.special.softmax(r_centers_0 + np.log(phi_k))).cumsum()

        delta_rui = np.matmul(self.user_embs[user_id], self.item_emb_res.T)
        combine_tmp = sp.sparse.csr_matrix((np.exp(delta_rui), (np.arange(self.num_items), np.arange(self.num_items))), shape=(self.num_items, self.num_items))
        combine_mat = combine_tmp * self.combine_cluster

        # combine_max = np.max(combine_mat, axis=0).A
        # combine_norm = 


        # w_kk : \sum_{i\in K K'} exp(rui)
        w_kk = combine_mat.sum(axis=0).A
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        self.kk_mtx_res = w_kk.reshape((self.num_cluster, self.num_cluster))

    
    def __sampler__(self, user_id):
        def sample():
            seeds_values = np.random.rand(self.num_neg)
            sampled_cluster_0 = np.array(list(map(lambda x : bisect.bisect(self.p_table_0, x) , seeds_values)))
            p_0 = np.array(self.comput_p(sampled_cluster_0, self.p_table_0))


            seeds_values = np.random.rand(self.num_neg)
            sampled_cluster_1 = list(map(lambda idx, x: bisect.bisect(self.p_table_1[idx], x) , sampled_cluster_0, seeds_values))
            
            p_1 = np.squeeze(np.array([self.comput_p(np.array([idx_1]), self.p_table_1[idx_0]) for idx_1, idx_0  in zip(sampled_cluster_1, sampled_cluster_0)]))

            idx_final_cluster = [idx_0 * self.num_cluster + idx_1 for idx_0, idx_1 in zip(sampled_cluster_0, sampled_cluster_1)]

            idx_items_lst = [[ self.combine_cluster.indices[j] for j in range(self.combine_cluster.indptr[c], self.combine_cluster.indptr[c+1])] for c in idx_final_cluster]
            
            final_items = []
            final_probs = []
            for items in idx_items_lst:
                rui_items =  np.matmul(self.user_embs[user_id], self.item_emb_res[items].T)
                # to compute on the fly without delta
                item_sample_idx, p = self.sample_final_items(rui_items)
                final_items.append(items[item_sample_idx])
                final_probs.append(p)
            
            final_probs = np.log(p_0) + np.log(p_1) +  np.log(np.array(final_probs))
            return final_items, final_probs
        return sample

    def compute_item_p(self, user_id, item_list):
        clusters_idx = self.combine_cluster[item_list, :].nonzero()[1] # find the combine cluster idx
        k_0 = clusters_idx // self.num_cluster
        k_1 = clusters_idx % self.num_cluster
        
        p_0 = np.array(self.comput_p(k_0, self.p_table_0))
        p_1 = np.squeeze(np.array([self.comput_p(np.array([idx_1]), self.p_table_1[idx_0]) for idx_1, idx_0 in zip(k_1, k_0)]))

        frac = np.matmul(self.user_embs[user_id], self.item_emb_res[item_list].T)
        deno = np.array([self.kk_mtx_res[idx_0, idx_1] for idx_0, idx_1 in zip(k_0, k_1)])
        # p_r = np.exp(frac - deno)
        return np.log(p_0) + np.log(p_1) + frac - deno




def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start_user
    overall_end = dataset.end_user
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start_user = overall_start + worker_id * per_worker
    dataset.end_user = min(dataset.start_user + per_worker, overall_end)


def setup_seed(seed):
    import os
    os.environ['PYTHONHASHSEED']=str(seed)

    import random
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

from dataloader import RecData, Sampler_Dataset
from torch.utils.data import DataLoader
if __name__ == "__main__":
    data = RecData('datasets', 'ml100kdata.mat')
    train, test = data.get_data(0.8)
    user_num, item_num = train.shape
    # user_emb = np.load('u.npy')
    # item_emb = np.load('v.npy')
    setup_seed(20)
    user_emb, item_emb = torch.randn((user_num, 20)), torch.randn((item_num, 20))
    # user_emb , item_emb = torch.tensor(user_emb), torch.tensor(item_emb)
    sampler = ExactSamplerModel(train, 100000, user_emb, item_emb, 25)
    # sampler = ExactSamplerModel(train[:50], 5000, user_emb, item_emb, 25)

    train_sample = Sampler_Dataset(sampler)
    train_dataloader = DataLoader(train_sample, batch_size=16, num_workers=6,worker_init_fn=worker_init_fn)
    b = 0
    for idx, data in enumerate(train_dataloader):
        user_id, user_his, ruis, neg_id, prob = data
        # print('Batch ', idx)
        # if 0 not in user_id:
        #     continue

        uid = user_id[0]
        print(uid)

        # delta = sampler.item_emb_res
        # popular_pop = sampler.pop_count
        u_emb = user_emb[uid,:]
        # scores = torch.matmul(u_emb, (item_emb - delta).T).squeeze(dim=0)
        # scores = scores + torch.log(torch.tensor(popular_pop))
        scores = torch.matmul(u_emb, item_emb.T).squeeze(dim=0)
        probs = F.softmax(scores, dim=-1).numpy()

        
        prob_pos = ruis[0]
        # print(torch.tensor(probs) * user_his, prob_pos)
        # probs = np.cumsum(probs, axis=-1)
        # probs_str = ['%s : %.5f' % (i ,probs[i]) for i in range(len(probs))]
        # # print(' '.join(probs_str))
        # print('\n')

        # count_arr = np.zeros(item_num, np.float)
        # for item in neg_id[0]:
        #     count_arr[item] += 1
        # count_str =  ['%s : %d' % (i ,count_arr[i]) for i in range(len(count_arr))]
        # # print(' '.join(count_str))
        # print('\n')
        # count_arr = count_arr / np.sum(count_arr)
        # count_arr = np.cumsum(count_arr, axis=-1)
        # count_str =  ['%s : %.5f' % (i ,count_arr[i]) for i in range(len(count_arr))]
        # # print(' '.join(count_str))

        # import matplotlib.pyplot as plt
        # plt.plot(probs, linewidth = '2', label='computed softmax')
        # plt.plot(count_arr, label='sampled_dis')
        # plt.legend()
        # plt.savefig('dis_25_e_more.jpg')
        # import pdb; pdb.set_trace()
        b = b + 1
    print('finish')