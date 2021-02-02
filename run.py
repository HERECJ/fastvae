from dataloader import RecData, UserItemData, Sampler_Dataset
from sampler import SamplerUserModel, PopularSamplerModel, ExactSamplerModel, SoftmaxApprSampler, SoftmaxApprSamplerUniform, SoftmaxApprSamplerPop, UniformSoftmaxSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from vae_models import BaseVAE, VAE_Sampler
import argparse
import numpy as np
from utils import Eval
import utils
import logging
import scipy as sp
import scipy.io
import datetime
import time
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def evaluate(model, train_mat, test_mat, config, logger, device):
    logger.info("Start evaluation")
    model.eval()
    with torch.no_grad():
        user_num, item_num = train_mat.shape
        
        user_emb = get_user_embs(train_mat, model, device)
        item_emb = model._get_item_emb()
        
        user_emb = user_emb.cpu().data
        item_emb = item_emb.cpu().data
        # ratings = torch.matmul(user_emb, item_emb.T)
        users = np.random.choice(user_num, min(user_num, 5000), False)
        evals = Eval()
        m = evals.evaluate_item(train_mat[users, :], test_mat[users, :], user_emb[users, :], item_emb, topk=50)
    return m

def get_user_embs(data_mat, model, device):
    data = UserItemData(data_mat, train_flag=False)
    dataloader = DataLoader(data, batch_size=config.batch_size_u, num_workers=config.num_workers, pin_memory=True, shuffle=False, collate_fn=utils.custom_collate)
    user_lst = []
    for e in dataloader:
        user_his, _, _, _ = e
        user_emb = model._get_user_emb(user_his.to(device))
        user_lst.append(user_emb)
    return torch.cat(user_lst, dim=0)

def train_model(model, train_mat, test_mat, config, logger):
    sampler_list = [SamplerUserModel, PopularSamplerModel, ExactSamplerModel, SoftmaxApprSampler, SoftmaxApprSamplerUniform, SoftmaxApprSamplerPop, UniformSoftmaxSampler]
    optimizer = utils_optim(config.learning_rate, model)
    scheduler = StepLR(optimizer, config.step_size, config.gamma)
    device = torch.device(config.device)
    for epoch in range(config.epoch):
        loss_ , kld_loss = 0.0, 0.0
        logger.info("Epoch %d"%epoch)
        # print("--Epoch %d"%epoch)
        
        if config.sampler == 0:
            train_data = UserItemData(train_mat)
            train_dataloader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=True, collate_fn=utils.custom_collate)
            # logging.info('Finish Loading Dataset, Start training !!!')


        elif config.sampler > 0:
            user_emb = get_user_embs(train_mat, model, device)
            item_emb = model._get_item_emb()

            sampler = sampler_list[config.sampler-1](train_mat, config.sample_num, user_emb, item_emb, config.cluster_num)
            train_data = Sampler_Dataset(sampler)
            train_dataloader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers,worker_init_fn=utils.worker_init_fn, collate_fn=utils.custom_collate, pin_memory=True)        
            # logging.info('Finish Sampling, Start training !!!')
        
        for batch_idx, data in enumerate(train_dataloader):
            model.train()

            pos_id, prob_pos, neg_id, prob_neg = data
            pos_id, prob_pos, neg_id, prob_neg = pos_id.to(device), prob_pos.to(device), neg_id.to(device), prob_neg.to(device)
            optimizer.zero_grad()
            pos_rat, neg_rat, mu, logvar = model(pos_id, neg_id) 

            loss = model.loss_function(neg_rat, prob_neg, pos_rat, prob_pos, reduction=config.reduction)
            
            kl_divergence = model.kl_loss(mu, logvar, config.anneal, reduction=config.reduction)/config.batch_size

            # if (batch_idx % 5) == 0:
            # logger.info("--Batch %d, loss : %.4f, kl_loss : %.4f "%(batch_idx, loss.data, kl_divergence))
            loss_ += loss
            kld_loss += kl_divergence
            loss += kl_divergence
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
        
        logger.info('--loss : %.2f, kl_dis : %.2f, total : %.2f '% (loss_, kld_loss, loss_ + kld_loss))
            
        scheduler.step()
        if (epoch % 5) == 0:
            result = evaluate(model, train_mat, test_mat, config, logger, device)
            logger.info('***************Eval_Res : NDCG@5,10,50 %.6f, %.6f, %.6f'%(result['item_ndcg'][4], result['item_ndcg'][9], result['item_ndcg'][49]))
            logger.info('***************Eval_Res : RECALL@5,10,50 %.6f, %.6f, %.6f'%(result['item_recall'][4], result['item_recall'][9], result['item_recall'][49]))


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

    assert config.sample_num < item_num
    
    if config.model == 'vae' and config.sampler == 0:
        model = BaseVAE(item_num, config.dim)
    elif config.model == 'vae' and config.sampler > 0:
        model = VAE_Sampler(item_num, config.dim)
    else:
        raise ValueError('Not supported model name!!!')
    model = model.to(device)
    train_model(model, train_mat, test_mat, config, logger)
    
    return evaluate(model, train_mat, test_mat, config, logger, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('-data', default='ml10M', type=str, help='path of datafile')
    parser.add_argument('-d', '--dim', default=[200, 32], type=list, help='the dimenson of the latent vector for student model')
    parser.add_argument('-s','--sample_num', default=500, type=int, help='the number of sampled items')
    parser.add_argument('--subspace_num', default=2, type=int, help='the number of splitted sub space')
    parser.add_argument('--cluster_num', default=16, type=int, help='the number of cluster centroids')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='the batch size for training')
    parser.add_argument('-e','--epoch', default=50, type=int, help='the number of epoches')
    parser.add_argument('-o','--optim', default='adam', type=str, help='the optimizer for training')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='the learning rate for training')
    parser.add_argument('--seed', default=20, type=int, help='random seed values')
    parser.add_argument('--ratio', default=0.8, type=float, help='the spilit ratio of dataset for train and test')
    parser.add_argument('--log_path', default='logs_test', type=str, help='the path for log files')
    parser.add_argument('--num_workers', default=8, type=int, help='the number of workers for dataloader')
    parser.add_argument('--data_dir', default='datasets', type=str, help='the dir of datafiles')
    parser.add_argument('--device', default='cuda', type=str, help='device for training, cuda or gpu')
    parser.add_argument('--model', default='vae', type=str, help='model name')
    parser.add_argument('--sampler', default=1, type=int, help='the sampler, 0 : no sampler, 1: uniform, 2: popular, 3: extrasoftmax, 4: ours, 5: our+uniform, 6: our+pop, 7: uniform+softmax')
    parser.add_argument('--fix_seed', default=True, type=bool, help='whether to fix the seed values')
    parser.add_argument('--step_size', default=5, type=int, help='step size for learning rate discount')
    parser.add_argument('--gamma', default=0.95, type=float, help='discout for lr')
    parser.add_argument('--anneal', default=1.0, type=float, help='parameters for kl loss')
    parser.add_argument('--batch_size_u', default=128, type=int, help='batch size user for inference')
    parser.add_argument('--reduction', default=False, type=bool, help='loss if reduction')


    config = parser.parse_args()

    import os
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    
    alg = config.model
    sampler = str(config.sampler)
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((config.data, alg, sampler, timestamp))
    log_file_name = os.path.join(config.log_path, loglogs)
    logger = utils.get_logger(log_file_name)
    
    logger.info(config)
    if config.fix_seed:
        utils.setup_seed(config.seed)
    m = main(config, logger)
    # print('ndcg@5,10,50, ', m['item_ndcg'][[4,9,49]])

    logger.info('Eval_Res : NDCG@5,10,50 %.6f, %.6f, %.6f'%(m['item_ndcg'][4], m['item_ndcg'][9], m['item_ndcg'][49]))
    logger.info('Eval_Res : RECALL@5,10,50 %.6f, %.6f, %.6f'%(m['item_recall'][4], m['item_recall'][9], m['item_recall'][49]))

    logger.info("Finish")
    svmat_name = log_file_name + '.mat'
    scipy.io.savemat(svmat_name, m)
