import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from vae_models import QVAE_CF, VAE_CF, Base_VAECF
import argparse
from base_dataloader import RecData, UserItemData
import numpy as np
from utils import Eval
import logging, coloredlogs
import scipy as sp
import scipy.io
import datetime, time
import math
# coloredlogs.install(level='DEBUG')

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

def compute_loss(pos_ratings, partition_ratings):
    import ipdb; ipdb.set_trace()
    return (-pos_ratings + torch.log(torch.sum(torch.exp(partition_ratings), dim=-1))).mean()



def evaluate(model, train_mat, test_mat, config, logger):
    logger.info("Start evaluation")
    model.eval()
    with torch.no_grad():
        user_num, item_num = train_mat.shape
        user_emb, item_emb = model.get_uv()
        # ratings = torch.matmul(user_emb, item_emb.T)
        users = np.random.choice(user_num, min(user_num, 50000), False)
        evals = Eval()
        m = evals.evaluate_item(train_mat[users, :], test_mat[users, :], user_emb[users, :], item_emb, topk=-1)
    
    return m




def train_model(model, train_mat, test_mat, config, logger):
    optimizer = utils_optim(config, model)
    for epoch in range(config.epoch):
        loss = 0
        logger.info("Epoch %d, Start Sampling !!!"%epoch)
        # print("--Epoch %d"%epoch)

        train_data = UserItemData(train_mat)
        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers)
        
        
        for batch_idx, data in enumerate(train_dataloader):
            model.train()
            user_id, user_history = data
            optimizer.zero_grad()
            item_logits = model(user_id, user_history)
            loss = item_logits.sum(-1).mean()
            kl_user, kl_item = model.klv_loss()
            # kl_divergence = kl_user / config.batch_size + kl_item / config.batch_size
            kl_divergence = (kl_user + kl_item) / (batch_idx + 1.0)
            loss += kl_divergence
            loss.backward()
            optimizer.step()
            if (batch_idx % 20) == 0:
                logger.info("--Batch %d, loss : %.4f, kl_loss : %.4f "%(batch_idx, loss.data, kl_divergence))
        
        if (epoch % 5) == 0:
            result = evaluate(model, train_mat, test_mat, config, logger)
            logger.info('*********===========NDCG@5,10,50 %.6f, %.6f, %.6f'%(result['item_ndcg'][4], result['item_ndcg'][9], result['item_ndcg'][49])) 



def utils_optim(config, model):
    if config.optim=='adam':
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optim=='sgd':
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError('Unkown optimizer!')
        
        

def main(config, user_q=False, logger=None):
    data = RecData(config.data_dir, config.data)
    train_mat, test_mat = data.get_data(config.ratio)
    user_num, item_num = train_mat.shape
    logging.info('The shape of datasets: %d, %d'%(user_num, item_num))
    
    latent_dim = config.dim
    num_subspace = config.subspace_num
    cluster_dim = config.cluster_dim
    res_dim = latent_dim - num_subspace * cluster_dim
    assert ( res_dim ) > 0
    

        # modify a real value vectors
    model = Base_VAECF(user_num, item_num, latent_dim)
    train_model(model, train_mat, test_mat, config, logger)
    
    return evaluate(model, train_mat, test_mat, config, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('-data', default='ml10Mdata.mat', type=str, help='path of datafile')
    parser.add_argument('-d', '--dim', default=64, type=int, help='the dimenson of the latent vector for student model')
    # parser.add_argument('-r', '--reg', default=1e-2, type=float, help='coefficient of the regularizer')
    parser.add_argument('-s','--sample_num', default=50, type=int, help='the number of sampled items')
    parser.add_argument('--subspace_num', default=2, type=int, help='the number of splitted sub space')
    parser.add_argument('--cluster_num', default=16, type=int, help='the number of cluster centroids')
    parser.add_argument('--cluster_dim', default=6, type=int, help=' the dimension of the cluster' )
    # parser.add_argument('--res_dim', default=0, type=int, help='residual dimension latent_dim - subspace_num * cluster_dim')
    parser.add_argument('--encode_subspace', default=2, type=int, help='the subspace for user encoding')
    parser.add_argument('--encode_cluster', default=48, type=int, help='the number of clusters for user encoding')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='the batch size for training')
    parser.add_argument('-e','--epoch', default=16, type=int, help='the number of epoches')
    parser.add_argument('-o','--optim', default='adam', type=str, help='the optimizer for training')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='the learning rate for training')
    parser.add_argument('--seed', default=20, type=int, help='random seed values')
    parser.add_argument('--ratio', default=0.8, type=float, help='the spilit ratio of dataset for train and test')
    parser.add_argument('--log_path', default='log_base', type=str, help='the path for log files')
    parser.add_argument('--user_quatized', default=False, type=bool, help='whether to quantize the user embeddings')
    parser.add_argument('--num_workers', default=8, type=int, help='the number of workers for dataloader')
    parser.add_argument('--data_dir', default='datasets', type=str, help='the dir of datafiles')


    config = parser.parse_args()

    import os
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    
    alg = 'vaecf'
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((config.data[:-4], alg, timestamp))
    log_file_name = os.path.join(config.log_path, loglogs)
    logger = get_logger(log_file_name)
    
    logger.info(config)
    setup_seed(config.seed)
    m = main(config, config.user_quatized, logger)
    print('ndcg@5,10,50, ', m['item_ndcg'][[4,9,49]])
    logger.info("Finish")
    svmat_name = log_file_name + '.mat'
    scipy.io.savemat(svmat_name, m)
    # import pdb; pdb.set_trace()
    # import cProfile
    # cProfile.run('main(config, config.user_quatized, logger)')
