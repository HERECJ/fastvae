import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader
from variational_autoencoder import QVAE_CF, VAE_CF
import argparse
from dataloader import RecData, UserItemData
from sampler import sampler
import pdb


def compute_loss(true_label, pos_ratings, partition_ratings):
    # for implicit feedbacks the value of true_label is 1
    prob_softmax = F.softmax(partition_ratings, dim=-1).detach()
    neg_rat = torch.sum(prob_softmax * partition_ratings,dim=-1)
    return (true_label * (pos_ratings - neg_rat)).sum()

def train_model(model, dataloader, config):
    optimizer = utils_optim(config, model)
    for epoch in range(config.epoch):
        model.train()
        loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            user_id, item_id = data
            optimizer.zero_grad()
            import pdb; pdb.set_trace()
            out_puts = model(user_id, item_id)
            import pdb; pdb.set_trace()
            kl_user, kl_item = model.klv_loss()
            kl_divergence = kl_user / config.batch_size + kl_item / config.batch_size
            import pdb; pdb.set_trace()
            optimizer.step()



def utils_optim(config, model):
    if config.optim=='adam':
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optim=='sgd':
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError('Unkown optimizer!')
        
        

def main(config, user_q=False):
    data = RecData(config.data)
    train_mat, test_mat = data.get_data()
    user_num, item_num = train_mat.shape
    latent_dim = config.dim
    num_subspace = config.subspace_num
    num_cluster = config.cluster_num
    cluster_dim = config.cluster_dim
    res_dim = latent_dim - num_subspace * cluster_dim
    assert ( res_dim ) > 0

    train_data = UserItemData(train_mat)
    train_dataloader = DataLoader(train_data,batch_size=config.batch_size,shuffle=True)

    
    if user_q:
        model = QVAE_CF(user_num, item_num, latent_dim, num_partitions=config.encode_subspace, num_centroids=config.encode_cluster)
    else:
        # modify a real value vectors
        model = VAE_CF(user_num, item_num, latent_dim)
    train_model(model, train_dataloader, config)
    


    
    # item_embeds = model.get_item_emb()
    # sample = sampler(item_embeds[:500], item_embeds, num_subspace, cluster_dim, num_cluster, res_dim)
    # sample.preprocess(10)
    # item = sample.__sampler__()



    print(user_num, item_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('-data', default='ml100kdata.mat', type=str, help='path of datafile')
    parser.add_argument('-d', '--dim', default=20, type=int, help='the dimenson of the latent vector for student model')
    parser.add_argument('-r', '--reg', default=1e-3, type=float, help='coefficient of the regularizer')
    parser.add_argument('-s','--sample_num', default=3, type=int, help='the number of sampled items')
    parser.add_argument('--subspace_num', default=2, type=int, help='the number of splitted sub space')
    parser.add_argument('--cluster_num', default=16, type=int, help='the number of cluster centroids')
    parser.add_argument('--cluster_dim', default=6, type=int, help=' the dimension of the cluster' )
    # parser.add_argument('--res_dim', default=0, type=int, help='residual dimension latent_dim - subspace_num * cluster_dim')
    parser.add_argument('--encode_subspace', default=2, type=int, help='the subspace for user encoding')
    parser.add_argument('--encode_cluster', default=8, type=int, help='the number of clusters for user encoding')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='the batch size for training')
    parser.add_argument('-e','--epoch', default=30, type=int, help='the number of epoches')
    parser.add_argument('-o','--optim', default='adam', type=str, help='the optimizer for training')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='the learning rate for training')

    config = parser.parse_args()
    print(config)
    main(config,user_q=True)
