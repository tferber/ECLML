import torch
from losses.losses_tools import create_loss_dict
from metrics.metric_tools import create_metric_dict

from torch_geometric.nn import global_add_pool


# fast clustering metrics: mean, sigma, fraction in range
def clustermetric(batch, pred, amin1=0.7, amax1=1.2, amin2=0.85, amax2=1.1):
    
    ldict = create_loss_dict(batch, pred)
    
    r_energy    = ldict['r_energy'] # (B x V) x T
    t_energy    = ldict['t_energy'] # (B x V) x T
    t_sigfrac   = ldict['t_sigfrac'] # (B x V) x T
    p_sigfrac   = ldict['p_sigfrac'] # (B x V) x T
    
    # calculate R_k numerator and denominator
    p_sum = torch.mul(torch.unsqueeze(r_energy, -1), p_sigfrac)
    t_sum = torch.mul(torch.unsqueeze(r_energy, -1), t_sigfrac)
    
    # sum over all nodes
    p_sum_k = global_add_pool(p_sum, batch.batch)
    t_sum_k = global_add_pool(t_sum, batch.batch)

    r = torch.where(t_sum_k>0, p_sum_k/t_sum_k, torch.zeros_like(t_sum_k))    
    a1 = ((r >= amin1) & (r < amax1)).sum(0).view(1,-1)
    a2 = ((r >= amin2) & (r < amax2)).sum(0).view(1,-1)
    
    return r, a1, a2 # dimension: B

def clustermetric_true(batch, pred):
    ldict = create_loss_dict(batch, pred)
    
    r_energy    = ldict['r_energy'] # (B x V) x T
    t_sigfrac   = ldict['t_sigfrac'] # (B x V) x T
    t_sum = torch.mul(torch.unsqueeze(r_energy, -1), t_sigfrac)
    t_sum_global = global_add_pool(t_sum, batch.batch)
    
    return t_sum_global

def clustermetric_pred(batch, pred):
    
    ldict = create_loss_dict(batch, pred)
    r_energy    = ldict['r_energy'] # (B x V) x T
    p_sigfrac   = ldict['p_sigfrac'] # (B x V) x T
    p_sum = torch.mul(torch.unsqueeze(r_energy, -1), p_sigfrac)
    p_sum_global = global_add_pool(p_sum, batch.batch)

    return p_sum_global
   
# clustering metric for baseline
def clustermetric_baseline(batch):
    
    mdict = create_metric_dict(batch)
    r_baseline    = mdict['r_baseline'] # (B x V) x T

    return r_baseline
    
    
    
    
    
    
