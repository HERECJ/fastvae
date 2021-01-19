import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pdb

class MF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(MF, self).__init__()
        self._User_Embedding = nn.Embedding(num_users, latent_dim)
        self._Item_Embedding = nn.Embedding(num_items, latent_dim)
    
    def forward(self, user_id, item_id):
        user_emb = self._User_Embedding(user_id)
        item_emb = self._Item_Embedding(item_id)
        return torch.sum(user_emb * item_emb, dim=-1)

def compute_loss1(ruis, pos_idx):
    exp_rui = torch.exp(ruis)
    pos_rat = torch.log(exp_rui[pos_idx])
    sum_exp = 0.0
    # import pdb; pdb.set_trace()
    for i in range(len(ruis)):
        if i == pos_idx:
            continue
        sum_exp += exp_rui[i]
    neg_rat = torch.log(sum_exp)
    # return pos_rat - neg_rat
    return neg_rat
 
def compute_loss2(ruis, pos_idx, pos_vec):
    # pos_rat = 0.5 * ruis[pos_idx]**2
    pos_rat = ruis[pos_idx]
    
    sum_neg = ruis * torch.tensor(pos_vec)
    probs = F.softmax(sum_neg).detach()
    # import pdb; pdb.set_trace()
    # logits = F.softmax(sum_neg)
    # import pdb; pdb.set_trace()
    neg_rat = (probs * sum_neg).sum()
    # neg_rat = sum_neg / (len(ruis) - 1)
    # return pos_rat - neg_rat
    return neg_rat

def main(num_users, num_items, latent_dim=16, data_len=10):
    model = MF(num_users, num_items, latent_dim)
    assert num_items >= data_len
    user_id = random.randint(0,num_users-1)
    user_tensor = torch.LongTensor([user_id for idx in range(data_len)])

    item_tensor = torch.LongTensor([i for i in range(data_len)])
    pos_idx = random.randint(0,data_len-1)
    pos_vec = [1 for i in range(data_len)]
    pos_vec[pos_idx] = -1e10

    ruis = model(user_tensor, item_tensor)
    # import pdb; pdb.set_trace()
    loss1 = compute_loss1(ruis, pos_idx)
    loss2 = compute_loss2(ruis, pos_idx, pos_vec)
    import pdb; pdb.set_trace()

    loss1.backward()
    import pdb; pdb.set_trace()
    for name, parms in model.named_parameters():
        print('loss1 \n-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad) 
    
    # loss2.backward()
    # # import pdb; pdb.set_trace()
    # for name, parms in model.named_parameters():
    #     print('\n\nloss2 \n-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
    return

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(20)
    main(5,10,data_len=6)