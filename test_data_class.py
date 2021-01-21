from torch.utils.data import Dataset, DataLoader
from sampler import SamplerModel
from dataloader import UserItemData, RecData
import torch


data = RecData('ml100kdata.mat')
train_mat, test_mat = data.get_data(0.8)
user_num, item_num = train_mat.shape
train_data = UserItemData(train_mat)

train_dataloader = DataLoader(train_data,32,shuffle=True)
# for x in train_dataloader:
    # user_id, item_id = x
    # print(user_id.shape, item_id.shape)

user_emb, item_emb = torch.rand((user_num, 20)), torch.rand((item_num, 20))
samples = SamplerModel(train_mat, user_emb, item_emb, 2, 8, 32)
# import pdb; pdb.set_trace()
gen_func = samples.negative_sampler(4)
# import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()
b = 1