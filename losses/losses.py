import torch
from losses_tools import create_loss_dict, energy_weighting
from torch_geometric.nn import global_add_pool

def frac_loss(batch, pred, usesqrt=True):
    
    # (B x V) is the number of vertices in the batch, T is the number of targets
    ldict = create_loss_dict(batch, pred)    
    t_energy    = ldict['t_energy'] # (B x V) x T
    t_sigfrac   = ldict['t_sigfrac'] # (B x V) x T
    p_sigfrac   = ldict['p_sigfrac'] # (B x V) x T
        
    # true energy weight: t_i^true * E_i^true.
    # for the last entry (==background), this is the reconstructed digit energy
    w_energy =  energy_weighting(t_energy, weight='sqrt')
    
    # step by step calculation 
    t_sum2 = global_add_pool(w_energy, batch.batch)
    diff_sum2 = global_add_pool(w_energy*(t_sigfrac - p_sigfrac)**2, batch.batch)
    ratio2 = diff_sum2/t_sum2
    sum2 = torch.sum(ratio2, dim=1) #sum over all targets
    loss = torch.mean(sum2, dim=0) #average over all graphs in batch
    
    return loss
