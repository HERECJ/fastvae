from torch.utils.data import IterableDataset, Dataset
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import scipy.io as sci
import scipy as sp
import random
import numpy as np
import math
import os

class RecData(object):
    def __init__(self, dir, file_name):
        file_name = file_name + 'data.mat'
        self.file_name = os.path.join(dir, file_name)

    def get_data(self,ratio):
        mat = self.load_file(file_name=self.file_name)
        train_mat, test_mat = self.split_matrix(mat, ratio)
        return train_mat, test_mat
    
    def load_file(self,file_name=''):
        if file_name.endswith('.mat'):
            return sci.loadmat(file_name)['data']
        else:
            raise ValueError('not supported file type')

    def split_matrix(self, mat, ratio=0.8):
        mat = mat.tocsr()  #按行读取，即每一行为一个用户
        m,n = mat.shape
        train_data_indices = []
        train_indptr = [0] * (m+1)
        test_data_indices = []
        test_indptr = [0] * (m+1)
        for i in range(m):
            row = [(mat.indices[j], mat.data[j]) for j in range(mat.indptr[i], mat.indptr[i+1])]
            train_idx = random.sample(range(len(row)), round(ratio * len(row)))
            train_binary_idx = np.full(len(row), False)
            train_binary_idx[train_idx] = True
            test_idx = (~train_binary_idx).nonzero()[0]
            for idx in train_idx:
                train_data_indices.append(row[idx]) 
            train_indptr[i+1] = len(train_data_indices)
            for idx in test_idx:
                test_data_indices.append(row[idx])
            test_indptr[i+1] = len(test_data_indices)

        [train_indices, train_data] = zip(*train_data_indices)
        [test_indices, test_data] = zip(*test_data_indices)

        train_mat = sp.sparse.csr_matrix((train_data, train_indices, train_indptr), (m,n))
        test_mat = sp.sparse.csr_matrix((test_data, test_indices, test_indptr), (m,n))
        return train_mat, test_mat


class UserItemData(Dataset):
    def __init__(self, train_mat):
        super(UserItemData, self).__init__()
        # self.train = train_mat.tocoo()
        # import pdb; pdb.set_trace()
        # self.user, self.item = self.train.row.astype(np.int64), self.train.col.astype(np.int64)
        self.train = train_mat
        self.users = np.random.permutation(self.train.shape[0])
    
    def __len__(self):
        return self.train.shape[0]
    
    def __getitem__(self, idx):
        # return self.user[idx], self.item[idx]
        pos_idx = self.train[self.users[idx]].nonzero()[1]
        return (pos_idx + 1).tolist(), 0, 0, 0

class Sampler_Dataset(IterableDataset):
    def __init__(self,sample_class):
        super(Sampler_Dataset, self).__init__()
        self.sampler = sample_class
        self.start_user = 0
        self.end_user = sample_class.num_users
    
    def __iter__(self):
        return self.sampler.negative_sampler(self.start_user, self.end_user)()
    

def get_max_length(x):
    return len(max(x, key=len))

def pad_sequence(seq):
    def _pad(_it, _max_len):
        return _it + [0] * (_max_len - len(_it))
    return [_pad(it, get_max_length(seq)) for it in seq]

def custom_collate(batch):
    transposed = zip(*batch)
    lst = []
    for samples in transposed:
        if type(samples[0]) in [np.int, np.int32, np.int64]:
               lst.append(torch.LongTensor(samples))
        else:
            if type(samples[0][0]) in [np.int, np.int32, np.int64]:
                lst.append(torch.LongTensor(pad_sequence(samples)))
            else:
                lst.append(torch.tensor(pad_sequence(samples)))
    return lst

if __name__ == "__main__":
    data = RecData('datasets', 'ml100k')
    train, test = data.get_data(0.8)
    train_data = UserItemData(train)
    train_loader = DataLoader(train_data, batch_size=10, num_workers=0, collate_fn=custom_collate)
    for idx, data in enumerate(train_loader):
        b = b+1