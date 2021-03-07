import torch
from losses_tools import create_loss_dict
from torch_geometric.nn import global_add_pool

def clustermetric(batch, pred, amin=0.7, amax=1.2):
    
    # need a dictionary as well
    ldict = create_loss_dict(batch, pred)
    
    r_energy    = ldict['r_energy'] # (B x V) x T
    t_energy    = ldict['t_energy'] # (B x V) x T
    t_sigfrac   = ldict['t_sigfrac'] # (B x V) x T
    p_sigfrac   = ldict['p_sigfrac'] # (B x V) x T
    
    # calculate R_k (k=1, 2, ..., bkgd)
    p_sum = torch.mul(torch.unsqueeze(r_energy, -1), p_sigfrac)
    t_sum = torch.mul(torch.unsqueeze(r_energy, -1), t_sigfrac)
    
    # sum over all nodes
    p_sum_k = global_add_pool(p_sum, batch.batch)
    t_sum_k = global_add_pool(t_sum, batch.batch)

    r = torch.where(t_sum_k>0, p_sum_k/t_sum_k, torch.zeros_like(t_sum_k))    
    a = ((r >= amin) & (r < amax)).sum(0).view(1,-1)
    
    return r, a # dimension: B