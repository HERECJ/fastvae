import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from vae_models import BaseVAE, VAE_CF, QVAE_CF
import argparse
from dataloader import RecData, UserItemData, Sampler_Dataset
from sampler2 import SamplerUserModel, PopularSamplerModel, ExactSamplerModel, SoftmaxApprSampler, SoftmaxApprSamplerUniform, SoftmaxApprSamplerPop, UniformSoftmaxSampler
import numpy as np
from utils import Eval
import logging, coloredlogs
import scipy as sp
import scipy.io
import datetime, time
import math
# coloredlogs.install(level='DEBUG')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_logger(filename, verbosity=1, name=None):
    filename = filename + '.txt'
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def setup_seed(seed):
    import os
    os.environ['PYTHONHASHSEED']=str(seed)

    import random
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



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

def compute_loss(user_his, prob_pos, pos_rats, part_rats, prob_neg=None,reduction=True, loss_mode=0):
    if loss_mode == 0 :
        user_his = user_his.squeeze(dim=1)
        item_logits = torch.negative(F.log_softmax(part_rats, dim=-1))
        scores = (item_logits[user_his>0]).sum(-1)
    elif loss_mode == 1:
        # Do not sub rated items from the sampled items
        # probs = F.softmax( part_rats - prob_neg, dim=-1).detach()
        # final = torch.sum(probs * part_rats).unsqueeze(-1)
        # scores = ((- pos_rats + final) * user_his).sum(-1)
        new_pos = pos_rats - torch.log(prob_pos.detach())
        new_neg = part_rats - torch.log(prob_neg.detach())
        final = torch.logsumexp(new_neg, dim=-1).unsqueeze(-1)
        scores = ((- new_pos + final)[user_his>0]).sum(-1)

    elif loss_mode == 2:
        # concat the rated items into the sampled items
        new_pos = pos_rats - torch.log(prob_pos.detach())
        new_neg = part_rats - torch.log(prob_neg.detach())
        # only_pos = new_pos * user_his - (1 - user_his) * 1e9
        new_pos[~(user_his>0)] = -float("Inf")
        expected_rat = torch.cat((new_pos, new_neg), dim=1)
        final = torch.logsumexp(expected_rat, dim=-1).unsqueeze(-1)
        scores =  ((- new_pos + final)[user_his>0]).sum(-1)
    
    elif loss_mode == 3:
        new_pos = pos_rats - torch.log(prob_pos.detach())
        new_neg = part_rats - torch.log(prob_neg.detach())
        parts_sum_exp = torch.sum(torch.exp(new_neg), dim=-1).unsqueeze(-1)
        new_pos[user_his<1] = -float("Inf")
        tmp = torch.exp(new_pos) + parts_sum_exp
        final = torch.log(tmp)
        scores =  ((- new_pos + final)[user_his>0]).sum(-1)


    if reduction:
        return scores.mean()
    else:
        return scores.sum()


def evaluate(model, train_mat, test_mat, config, logger):
    logger.info("Start evaluation")
    model.eval()
    with torch.no_grad():
        user_num, item_num = train_mat.shape
        user_emb, item_emb = model.get_uv()
        user_emb = user_emb.cpu().data
        item_emb = item_emb.cpu().data
        # ratings = torch.matmul(user_emb, item_emb.T)
        users = np.random.choice(user_num, min(user_num, 50000), False)
        evals = Eval()
        m = evals.evaluate_item(train_mat[users, :], test_mat[users, :], user_emb[users, :], item_emb, topk=200)
    return m


def train_model(model, train_mat, test_mat, config, logger):
    sampler_list = [SamplerUserModel, PopularSamplerModel, ExactSamplerModel, SoftmaxApprSampler, SoftmaxApprSamplerUniform, SoftmaxApprSamplerPop, UniformSoftmaxSampler]
    optimizer = utils_optim(config.learning_rate, model)
    lr = config.learning_rate
    device = torch.device(config.device)
    for epoch in range(config.epoch):
        loss_ = 0.0
        logger.info("Epoch %d, Start Sampling !!!"%epoch)
        # print("--Epoch %d"%epoch)
        
        if config.sampler == 0:
            train_data = UserItemData(train_mat)
            train_dataloader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
            logging.info('Finish Loading Dataset, Start training !!!')
            loss_mode = 0

        elif config.sampler > 0:
            user_emb, item_emb = model.get_uv()
            sampler = sampler_list[config.sampler-1](train_mat, config.sample_num, user_emb, item_emb, config.cluster_num)
            train_data = Sampler_Dataset(sampler)
            train_dataloader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers,worker_init_fn=worker_init_fn, pin_memory=True)        
            logging.info('Finish Sampling, Start training !!!')
            loss_mode = config.loss_mode
        
        for batch_idx, data in enumerate(train_dataloader):
            model.train()
            user_id, user_his, prob_pos, neg_id, prob_neg = data
            user_id, user_his, prob_pos, neg_id, prob_neg = user_id.to(device), user_his.to(device), prob_pos.to(device), neg_id.to(device), prob_neg.to(device)
            optimizer.zero_grad()
            pos_rats, part_rats = model(user_id, user_his, neg_id)
            # part_rats is the denominator of the softmax function
            loss = compute_loss(user_his, prob_pos, pos_rats, part_rats, prob_neg=prob_neg, loss_mode=loss_mode)
            # kl_divergence = model.klv_loss() / (batch_idx + 1.0)
            kl_divergence = model.klv_loss() / config.batch_size

            # if (batch_idx % 5) == 0:
            # logger.info("--Batch %d, loss : %.4f, kl_loss : %.4f "%(batch_idx, loss.data, kl_divergence))

            loss += kl_divergence
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_ += loss
        logger.info('-- loss : %.4f'% loss_)
            
        
        if (epoch % 20) == 0:
            result = evaluate(model, train_mat, test_mat, config, logger)
            logger.info('***************Eval_Res : NDCG@5,10,50 %.6f, %.6f, %.6f'%(result['item_ndcg'][4], result['item_ndcg'][9], result['item_ndcg'][49]))
        
        if (epoch % 5) == 0:
            lr = lr * 0.95
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    # user_emb, item_emb = model.get_uv()
    # np.save('u.npy', user_emb.cpu().detach().numpy())
    # np.save('v.npy', item_emb.cpu().detach().numpy())



def utils_optim(learning_rate, model):
    if config.optim=='adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif config.optim=='sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        raise ValueError('Unkown optimizer!')
        

def main(config, logger=None):
    device = torch.device(config.device)
    data = RecData(config.data_dir, config.data)
    train_mat, test_mat = data.get_data(config.ratio)
    user_num, item_num = train_mat.shape
    logging.info('The shape of datasets: %d, %d'%(user_num, item_num))
    
    latent_dim = config.dim
    num_subspace = config.subspace_num
    cluster_dim = config.cluster_dim
    res_dim = latent_dim - num_subspace * cluster_dim
    assert ( res_dim ) > 0
    assert config.sample_num < item_num
    
    if config.model == 'vae' and config.sampler == 0:
        model = BaseVAE(user_num, item_num, latent_dim)
    elif config.model == 'vae' and config.sampler > 0:
        model = VAE_CF(user_num, item_num, latent_dim)
    elif config.model == 'qvae':
        model = QVAE_CF(user_num, item_num, latent_dim, num_partitions=config.encode_subspace, num_centroids=config.encode_cluster)
    model = model.to(device)
    train_model(model, train_mat, test_mat, config, logger)
    
    return evaluate(model, train_mat, test_mat, config, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('-data', default='ml100kdata.mat', type=str, help='path of datafile')
    parser.add_argument('-d', '--dim', default=64, type=int, help='the dimenson of the latent vector for student model')
    # parser.add_argument('-r', '--reg', default=1e-2, type=float, help='coefficient of the regularizer')
    parser.add_argument('-s','--sample_num', default=100, type=int, help='the number of sampled items')
    parser.add_argument('--subspace_num', default=2, type=int, help='the number of splitted sub space')
    parser.add_argument('--cluster_num', default=16, type=int, help='the number of cluster centroids')
    parser.add_argument('--cluster_dim', default=6, type=int, help=' the dimension of the cluster' )
    # parser.add_argument('--res_dim', default=0, type=int, help='residual dimension latent_dim - subspace_num * cluster_dim')
    # parser.add_argument('--encode_subspace', default=2, type=int, help='the subspace for user encoding')
    # parser.add_argument('--encode_cluster', default=48, type=int, help='the number of clusters for user encoding')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='the batch size for training')
    parser.add_argument('-e','--epoch', default=60, type=int, help='the number of epoches')
    parser.add_argument('-o','--optim', default='adam', type=str, help='the optimizer for training')
    parser.add_argument('-lr', '--learning_rate', default=1e-1, type=float, help='the learning rate for training')
    parser.add_argument('--seed', default=20, type=int, help='random seed values')
    parser.add_argument('--ratio', default=0.8, type=float, help='the spilit ratio of dataset for train and test')
    parser.add_argument('--log_path', default='log_sampler', type=str, help='the path for log files')
    parser.add_argument('--num_workers', default=8, type=int, help='the number of workers for dataloader')
    parser.add_argument('--data_dir', default='datasets', type=str, help='the dir of datafiles')
    parser.add_argument('--device', default='cuda', type=str, help='device for training, cuda or gpu')
    parser.add_argument('--model', default='vae', type=str, help='model name')
    parser.add_argument('--sampler', default=4, type=int, help='the sampler, 0 : no sampler, 1: uniform, 2: popular, 3: extrasoftmax, 4: ours, 5: our+uniform, 6: our+pop, 7: uniform+softmax')
    parser.add_argument('--loss_mode', default=3, type=int, help='the loss mode for sampled items')
    parser.add_argument('--fix_seed', default=True, type=bool, help='whether to fix the seed values')


    config = parser.parse_args()

    import os
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    
    alg = config.model
    sampler = str(config.sampler)
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((config.data[:-4], alg, sampler, timestamp))
    log_file_name = os.path.join(config.log_path, loglogs)
    logger = get_logger(log_file_name)
    
    logger.info(config)
    if config.fix_seed:
        setup_seed(config.seed)
    m = main(config, logger)
    # print('ndcg@5,10,50, ', m['item_ndcg'][[4,9,49]])

    logger.info('Eval_Res : NDCG@5,10,50 %.6f, %.6f, %.6f'%(m['item_ndcg'][4], m['item_ndcg'][9], m['item_ndcg'][49]))

    logger.info("Finish")
    svmat_name = log_file_name + '.mat'
    scipy.io.savemat(svmat_name, m)