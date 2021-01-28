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
    def __init__(self, mat, num_neg, user_embs, item_embs, num_subspace, cluster_dim, num_cluster):
        self.mat = mat.tocsr()
        self.num_users, self.num_items = mat.shape
        self.num_neg = num_neg
        self.user_embs = user_embs = user_embs.cpu().data
        self.item_embs = item_embs = item_embs.cpu().data

    def preprocess(self, user_id):
        self.exist = set(self.mat.indices[j] for j in range(self.mat.indptr[user_id], self.mat.indptr[user_id + 1]))

    def __sampler__(self, user_id):
        def sample():
            return list(torch.utils.data.RandomSampler(range(self.num_items), num_samples=self.num_neg, replacement=True)), [1.0/self.num_items for _ in range(self.num_neg)]
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
                # ruis = torch.matmul(uid_emb, self.item_embs.T) * user_his
                yield i, user_his, torch.zeros(self.num_items), torch.LongTensor(neg_item), torch.tensor(prob)
        return generate_tuples

class PopularSamplerModel(SamplerUserModel):
    def __init__(self, mat, num_neg, user_embs, item_embs, num_subspace, cluster_dim, num_cluster, mode=0):
        super(PopularSamplerModel, self).__init__(mat, num_neg, user_embs, item_embs, num_subspace, cluster_dim, num_cluster,)
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
                p = self.pop_prob[k]
                neg_items.append(k)
                probs.append(p)
            return neg_items, probs
        return sample

class SoftmaxApprSampler(SamplerUserModel):
    """
    PQ methods, each item vector is splitted into three parts
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_subspace, cluster_dim, num_cluster, split='res'):
        super(SoftmaxApprSampler, self).__init__( mat, num_neg, user_embs, item_embs, num_subspace, cluster_dim, num_cluster)
        self.center_scores = {}
        self.combine_cluster_idx = torch.zeros((self.num_items))
        self.num_cluster = num_cluster
        self.num_subspace = num_subspace
        _, latent_dim = user_embs.shape
        if split == 'res':
            cluster_dim = latent_dim // num_subspace
        else:
            assert (latent_dim - num_subspace * cluster_dim) > 0
        
        # print(self.num_items, self.num_items)
        user_embs = user_embs.cpu().data
        item_embs = item_embs.cpu().data

        all_centers, all_codes = [],[]
        for i in range(self.num_subspace):
            start_idx = i * cluster_dim
            end_idx = (i+1) * cluster_dim
            cluster_kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(item_embs[:,start_idx:end_idx].detach().numpy())
            centers = cluster_kmeans.cluster_centers_
            codes = cluster_kmeans.labels_
            all_centers.append(centers)
            all_codes.append(codes)

            self.center_scores[i] = torch.matmul(user_embs[:,start_idx:end_idx] , torch.tensor(centers).T)
            self.combine_cluster_idx = self.combine_cluster_idx * self.num_cluster + codes
        
        if split == 'cut':
            self.delta_ruis = torch.matmul(user_embs[:,end_idx:], item_embs[:,end_idx:].T)
        else:
            assert (latent_dim - num_subspace * cluster_dim) == 0
            cluster_emb = [x[y] for x, y in zip(all_centers, all_codes)]
            cluster_emb = np.concatenate(cluster_emb, axis=1)
            item_emb_res = item_embs - cluster_emb
            self.delta_ruis = torch.matmul(user_embs, item_emb_res.T)
    
    def preprocess(self, user_id):
        self.exist = set(self.mat.indices[j] + 1 for j in range(self.mat.indptr[user_id], self.mat.indptr[user_id + 1]))

        self.user_id = user_id
        delta_ruis_u = torch.exp(self.delta_ruis[user_id]).detach().numpy()
        combine_mat = sp.sparse.csr_matrix((delta_ruis_u, (np.arange(self.num_items), self.combine_cluster_idx)), shape=(self.num_items, self.num_cluster**self.num_subspace))

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
            # print('!!!!!!  ',i)
            kk_mtx = np.matmul(kk_mtx, r_centers).squeeze(-1)
            self.sample_prob[i] = torch.tensor(kk_mtx)
    

    def sample_gumbel_noise(self, inputs,eps=1e-7, start_flag=False):
        if start_flag:
            us = torch.rand(inputs.unsqueeze(-1).expand(-1, self.num_neg).shape)
        else:
            us = torch.rand(inputs.shape)
        return -torch.log(- torch.log(us + eps) + eps)
    
    def sample_from_gumbel_noise(self, scores, start_flag=False):
        if start_flag:
            tmp = scores.unsqueeze(-1) + self.sample_gumbel_noise(scores, start_flag=True)
            max_class = torch.max(tmp, dim=0)
            # return max_class.indices, torch.exp(scores[max_class.indices]) * torch.exp(torch.negative(max_class.values))
            return max_class.indices, tmp[max_class.indices]
        else:
            tmp = scores + self.sample_gumbel_noise(scores)
            max_class = torch.max(tmp, dim=-1)
            # return max_class.indices, (torch.exp(torch.gather(scores, 1, max_class.indices.unsqueeze(0))) * torch.exp(torch.negative(max_class.values))).squeeze(dim=0)
            return max_class.indices, torch.gather(tmp, 1, max_class.indices.unsqueeze(0))

    
    def sample_from_weighted(self, scores, start_flag=False):
        probs = F.softmax(scores, dim=-1)
        if start_flag is True:
            sampled_items = torch.multinomial(probs, self.num_neg, replacement=True)
            # return sampled_items, probs[sampled_items]
            return sampled_items, scores[sampled_items]
        else:
            sampled_items = torch.multinomial(probs, 1, replacement=True)
            # return sampled_items.squeeze(-1), torch.gather(probs, 1, sampled_items).squeeze(-1)
            return sampled_items.squeeze(-1), torch.gather(scores, 1, sampled_items).squeeze(-1)
            

    def sample_func(self, scores, mode=0, start_flag=False, eps=1e-8):
        if mode == 0 :
            return self.sample_from_weighted(scores, start_flag)
        elif mode == 1:
            # Gambel Noise -> Approximate the softmax logits
            return self.sample_from_gumbel_noise(scores, start_flag)
        else:
            raise NotImplementedError('Not supported mode for sample function')
    
    def sample_final_items(self, scores, eps=1e-8):
        us = torch.rand(scores.shape)
        tmp = scores - torch.log(- torch.log(us + eps) + eps)
        max_class = torch.max(tmp, dim=0)
        # return max_class.indices, torch.exp(scores[max_class.indices]) * torch.exp(torch.negative(max_class.values))
        return max_class.indices, tmp[max_class.indices]

        

    def __sampler__(self, user_id):
        def sample():
            idx = []
            probs = []
            start_flag = True
            for i in range(self.num_subspace):
                sample_probs = self.sample_prob[i]
                if len(idx) > 0:
                    start_flag = False
                    for history_cluster in idx:
                        sample_probs = sample_probs[history_cluster]
                extra = sample_probs.squeeze()
                total_score = self.center_scores[i][self.user_id] + torch.log(extra)
                idx_clusters, _ = self.sample_func(total_score, start_flag=start_flag)
                idx.append(idx_clusters)
                p = self.center_scores[i][self.user_id][idx_clusters]
                probs.append(p)

            # sample from the final items
            # fprobs = torch.mul(probs[0], probs[1])
            fprobs = probs[0] +  probs[1]
            items = []
            final_probs = []
            i = 0
            while True:
                index_combine_cluster = idx[0][i] * self.num_cluster + idx[1][i]
                items_index = self.items_in_combine_cluster[index_combine_cluster]
                if len(items_index) == 1:
                    items.append(items_index[0])
                    final_probs.append(fprobs[i])
                elif len(items_index) < 1:
                    continue
                else:
                    rui_items = self.delta_ruis[self.user_id][items_index]
                    item_sample_index, _ = self.sample_final_items(rui_items)
                    items.append(items_index[item_sample_index])
                    p = rui_items[item_sample_index]
                    final_probs.append(fprobs[i] + p)

                i += 1
                if i > (self.num_neg-1):
                    break
            return items, final_probs
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
                uid_emb = self.user_embs[i]
                user_his = torch.tensor(self.mat[i].toarray()[0].tolist())
                ruis = torch.matmul(uid_emb, self.item_embs.T) * user_his
                yield i, user_his, ruis, torch.LongTensor(neg_item), torch.tensor(prob)
        return generate_tuples


# class ExactSamplerModel(SamplerUserModel):
#     def __init__(self, mat, model):
#         super(ExactSamplerModel, self).__init__(mat)
#         if isinstance(model, str):
#             model = np.load(model)
#         self.user = model['U']
#         self.item = model['V']
#     def preprocess(self, user_id):
#         super(ExactSamplerModel, self).preprocess(user_id)
#         pred = self.user[user_id] @ self.item.T
#         idx = np.argpartition(pred, -10)[-10:]
#         pred[idx] = -np.inf
#         self.score = sp.special.softmax(pred)
#         self.score_cum = self.score.cumsum()
#         self.score_cum[-1] = 1.0
#     def __sampler__(self, user_id, pos_id):
#         def sample():
#             k = bisect.bisect(self.score_cum, random.random())
#             p = self.score[k]
#             return k, p
#         return sample, self.exist

# class ClusterSamplerModel(SamplerModel):
#     def __init__(self, mat, model, num_clusters=100):
#         super(ClusterSamplerModel, self).__init__(mat)
#         if isinstance(model, str):
#             model = np.load(model)
#         user = model['U']
#         item = model['V']
#         clustering = KMeans(num_clusters, random_state=0).fit(item)
#         self.code = clustering.labels_
#         center = clustering.cluster_centers_
#         code_mat = sp.sparse.csr_matrix((np.ones_like(self.code), (np.arange(self.num_items), self.code)),
#                                         shape=(self.num_items, num_clusters))
#         cluster_num = np.squeeze(code_mat.sum(axis=0).A)
#         idx = cluster_num > 0
#         code_mat = code_mat[:, idx].tocsc()
#         self.code1 = code_mat.nonzero()[1]
#         self.num_clusters = code_mat.shape[1]
#         self.items_in_cluster = [[code_mat.indices[j] for j in range(code_mat.indptr[c], code_mat.indptr[c + 1])]
#                                  for c in range(self.num_clusters)]
#         self.num_items_in_cluster = np.array([len(self.items_in_cluster[c]) for c in range(self.num_clusters)])
#         self.center_score = np.exp(np.matmul(user, center[idx].T))

#     def preprocess(self, user_id):
#         super(ClusterSamplerModel, self).preprocess(user_id)
#         self.exist_items_in_cluster = [[] for _ in range(self.num_clusters)]
#         for e in self.exist:
#             self.exist_items_in_cluster[self.code[e]].append(e)
#         self.num_exist_items_in_cluster = np.array(
#             [len(self.exist_items_in_cluster[c]) for c in range(self.num_clusters)])

#         cs_user = self.center_score[user_id] * (self.num_items_in_cluster - self.num_exist_items_in_cluster)
#         self.cs_user = cs_user / cs_user.sum()
#         self.cs_user_cum = self.cs_user.cumsum()

#     def __sampler__(self, user_id, pos_id):
#         c = bisect.bisect(self.cs_user_cum, random.random())
#         prob = self.cs_user[c]/(self.num_items_in_cluster[c] - self.num_exist_items_in_cluster[c])
#         item = self.items_in_cluster[c]

#         def sample():
#             k = np.random.choice(self.num_items_in_cluster[c])
#             return item[k], prob

#         return sample, self.exist_items_in_cluster[c]

# class ClusterPopularSamplerModel(PopularSamplerModel):
#     def __init__(self, mat, model, num_clusters=10, **kwargs):
#         super(ClusterPopularSamplerModel, self).__init__(mat, **kwargs)
#         if isinstance(model, str):
#             model = np.load(model)
#         user = model['U']
#         if 'code' in model and 'center' in model:
#             center = model['center']
#             self.code = model['code']
#             num_clusters = center.shape[0]
#         else:
#             item = model['V']
#             clustering = KMeans(num_clusters, random_state=0).fit(item)
#             self.code = clustering.labels_
#             center = clustering.cluster_centers_

#         code_mat = sp.sparse.csr_matrix((np.ones_like(self.code), (np.arange(self.num_items), self.code)),
#                                         shape=(self.num_items, num_clusters))
#         cluster_num = np.squeeze(code_mat.sum(axis=0).A)
#         idx = cluster_num > 0
#         code_mat = code_mat[:, idx].tocsc()
#         self.code = code_mat.nonzero()[1]
#         self.num_clusters = code_mat.shape[1]
#         self.items_in_cluster = [np.array([code_mat.indices[j] for j in range(code_mat.indptr[c], code_mat.indptr[c + 1])])
#                                  for c in range(self.num_clusters)]
#         self.prob_items_in_cluster = [self.pop_prob[self.items_in_cluster[c]] for c in range(self.num_clusters)]
#         self.prob_cluster = np.array([np.sum(self.prob_items_in_cluster[c]) for c in range(self.num_clusters)])
#         self.prob_cum_items_in_cluster = [np.cumsum(self.prob_items_in_cluster[c])/self.prob_cluster[c] for c in range(self.num_clusters)]
#         self.center_score = np.exp(np.matmul(user, center[idx].T))

#     def preprocess(self, user_id):
#         super(ClusterPopularSamplerModel, self).preprocess(user_id)
#         self.exist_items_in_cluster = [[] for _ in range(self.num_clusters)]
#         for e in self.exist:
#             self.exist_items_in_cluster[self.code[e]].append(e)

#         self.prob_exist_items_in_cluster = [self.pop_prob[self.exist_items_in_cluster[c]] for c in range(self.num_clusters)]
#         self.prob_exist_cluster = np.array(
#             [np.sum(self.prob_exist_items_in_cluster[c]) for c in range(self.num_clusters)])

#         cs_user = self.center_score[user_id] * (self.prob_cluster - self.prob_exist_cluster)
#         self.cs_user = cs_user / cs_user.sum()
#         self.cs_user_cum = self.cs_user.cumsum()

#     def __sampler__(self, user_id, pos_id):
#         c = bisect.bisect(self.cs_user_cum, random.random())
#         prob = self.cs_user[c]/(self.prob_cluster[c] - self.prob_exist_cluster[c])
#         item = self.items_in_cluster[c]

#         def sample():
#             k = bisect.bisect(self.prob_cum_items_in_cluster[c], random.random())
#             return item[k], prob * self.pop_prob[item[k]]

#         return sample, self.exist_items_in_cluster[c]


# class TreeSamplerModel(PopularSamplerModel):
#     def __init__(self, mat, model, max_depth=10):
#         super(TreeSamplerModel, self).__init__(mat)
#         if isinstance(model, str):
#             model = np.load(model)
#         user = model['U']
#         item = model['V']
#         self.cluster_center, self.weight_sum_in_clusters, self.label = clustering.hierarchical_clustering(item, max_depth, weight=self.pop_prob)
#         max_depth = int(np.log2(self.cluster_center.shape[0] - 1))
#         self.leaf_start = 2**max_depth
#         self.num_leaves = self.cluster_center.shape[0] - self.leaf_start
#         self.items_in_leaf = clustering.distribute_into_leaf(np.arange(self.num_items), self.label, self.leaf_start, self.num_leaves)
#         self.prob_items_in_leaf = [self.pop_prob[self.items_in_leaf[c]] for c in range(self.num_leaves)]
#         self.prob_cum_items_in_leaf = [np.cumsum(self.prob_items_in_leaf[c])/self.weight_sum_in_clusters[c+self.leaf_start] for c in range(self.num_leaves)]
#         self.center_score = np.exp(np.matmul(user, self.cluster_center.T))

#     def preprocess(self, user_id):
#         super(TreeSamplerModel, self).preprocess(user_id)
#         self.label2weight = clustering.distribute_into_tree(self.exist, self.label, len(self.weight_sum_in_clusters), self.pop_prob)
#         self.exist_items_in_leaf = clustering.distribute_into_leaf(self.exist, self.label, self.leaf_start, self.num_leaves)
#         self.cs_user = self.center_score[user_id] * (self.weight_sum_in_clusters - self.label2weight)
#         self.cs_user[2::2] = self.cs_user[2::2] / (self.cs_user[2::2] + self.cs_user[3::2])
#         #for i in range(2, self.cs_user.shape[0], 2):
#         #    self.cs_user[i] = self.cs_user[i] / (self.cs_user[i+1] + self.cs_user[i])
#         #self.cs_user[3::2] = 1 - self.cs_user[2::2]

#     def __sampler__(self, user_id, pos_id):
#         c, p = clustering.leaf_sampling(self.cs_user)
#         #c, p = clustering.leaf_sampling(self.center_score[user_id], self.weight_sum_in_clusters, self.label2weight)
#         cum_prob = self.prob_cum_items_in_leaf[c - self.leaf_start]
#         prob = self.prob_items_in_leaf[c - self.leaf_start]
#         item_ = self.items_in_leaf[c - self.leaf_start]
#         exist_ = self.exist_items_in_leaf[c - self.leaf_start]

#         def sample():
#             k = bisect.bisect(cum_prob, random.random())
#             return item_[k], p * prob[k]

#         return sample, exist_

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

from dataloader import RecData, Sampler_Dataset
from torch.utils.data import DataLoader
if __name__ == "__main__":
    data = RecData('datasets', 'ml100kdata.mat')
    train, test = data.get_data(0.8)
    user_num, item_num = train.shape

    user_emb, item_emb = torch.rand((user_num, 20)), torch.rand((item_num, 20))
    # sampler = SoftmaxApprSampler(train[:1], 100000, user_emb, item_emb, 2, 6, 16)
    sampler = SamplerUserModel(train[:1], 100000, user_emb, item_emb, 2, 6, 16)

    train_sample = Sampler_Dataset(sampler)
    train_dataloader = DataLoader(train_sample, batch_size=16, num_workers=8,worker_init_fn=worker_init_fn)
    b = 0
    for idx, data in enumerate(train_dataloader):
        user_id, user_his, ruis, neg_id, prob = data

        # uid = user_id[0]
        # u_emb = user_emb[uid,:]
        # scores = torch.matmul(u_emb, item_emb.T).squeeze(dim=0)
        # probs = F.softmax(scores, dim=-1).numpy()
        # probs_str = ['%s : %.5f' % (i ,probs[i]) for i in range(len(probs))]
        # print(' '.join(probs_str))
        # print('\n')

        # count_arr = np.zeros(item_num, np.float)
        # for item in neg_id[0]:
        #     count_arr[item] += 1
        # count_str =  ['%s : %d' % (i ,count_arr[i]) for i in range(len(count_arr))]
        # print(' '.join(count_str))
        # print('\n')
        # count_arr = count_arr / np.sum(count_arr)
        # count_str =  ['%s : %.5f' % (i ,count_arr[i]) for i in range(len(count_arr))]
        # print(' '.join(count_str))
        # import pdb; pdb.set_trace()
        b = b + 1