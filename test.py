import torch 
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

# embeddings = nn.Embedding(50,128)
# num_subspace = 4
# item_id_array = torch.LongTensor(list(range(50)))
# item_emb = embeddings(item_id_array)
# vecs = item_emb.chunk(num_subspace, dim=-1)
# # pdb.set_trace()
# clusters = {}
# for idx, item in enumerate(vecs):
#     clusters[idx] = item
# # print(clusters)
# print(len(clusters), clusters[0].shape)
# pdb.set_trace()

# embs = nn.Embedding(40,32)
# user_emb = embs(torch.LongTensor([2,1,5,6]))
# print(user_emb.shape, clusters[1].T.shape)

# tt1 = torch.matmul(user_emb, clusters[1].T)
# tt2 = torch.matmul(user_emb, clusters[2].T)
# print(tt1.shape, tt2.shape)
# res1 = F.softmax(tt1)
# res2 = F.softmax(tt2)
# # print(tt, res)
# pdb.set_trace()
#  =====================================================
# seeds = torch.rand(4,10,20)
# N, K = seeds[0].shape
# for u in range(N):
#     tmp = seeds[:, u, :]
#     # idx_space = list(range(4))
#     res  = 1
    
#     for idx in range(4):
#         idx_vec = [1 for i in range(4)]
#         idx_vec[idx] = 20
#         vec = tmp[idx].view(idx_vec)
#         res = res * vec
#     # kk = torch.cartesian_prod(tmp[idx_space[0]], tmp)
#     pdb.set_trace()
#     print(res)
# =======================================================

from sklearn.cluster import KMeans

# a = np.random.random((200,20))
a = torch.randn((200,20))
clusters = KMeans(64,random_state=0).fit(a)
import pdb; pdb.set_trace()