import torch
from losses.losses_tools import create_loss_dict, energy_weighting, core_weighting
from torch_geometric.nn import global_add_pool



def frac_loss(batch, pred, usesqrt=True, weightcore=True, losspernode=False):
    
    # (B x V) is the number of vertices in the batch, T is the number of targets
    ldict = create_loss_dict(batch, pred)    
    r_theta  = ldict['r_theta'] # (B x V)
    r_phi    = ldict['r_phi'] # (B x V)
    r_energy = ldict['r_energy'] # (B x V)
        
    t_energy    = ldict['t_energy'] # (B x V) x T
    t_sigfrac   = ldict['t_sigfrac'] # (B x V) x T
    p_sigfrac   = ldict['p_sigfrac'] # (B x V) x T
            
    # true energy weight: t_i^true * E_i^true.
    # for the last entry (==background), this is the reconstructed digit energy
    w_energy =  energy_weighting(t_energy, usesqrt=usesqrt)
    
    # step by step calculation 
    t_sum2 = global_add_pool(w_energy, batch.batch)
    
    # add distance penalty
#     if weightcore:
#         cw = core_weighting(r_theta, r_phi, p_sigfrac, r_energy, batch.batch)
#     else:
#         cw = torch.ones_like(pred)

#     diff_sum2 = global_add_pool(w_energy*cw*(t_sigfrac - p_sigfrac)**2, batch.batch)
#     diff_sum2 = global_add_pool(w_energy*(t_sigfrac - p_sigfrac)**2, batch.batch)
#     diff_sum2 = global_add_pool(w_energy*((t_sigfrac - p_sigfrac)**2 + torch.maximum((t_sigfrac - p_sigfrac)**3, torch.zeros_like(t_sigfrac))), batch.batch)
    diff_sum2 = global_add_pool(w_energy*((t_sigfrac - p_sigfrac)**2 + torch.maximum((t_sigfrac - p_sigfrac)*0.05, torch.zeros_like(t_sigfrac))), batch.batch)
    
    #FIXME: only add positive penalty for physics targets?
    
    ratio2 = diff_sum2/t_sum2 # <-- what if tsum2 is 0?
    sum2 = torch.sum(ratio2, dim=1) #sum over all targets
    loss = torch.mean(sum2, dim=0) #average over all graphs in batch
    
#     # debug testing for nan
#     if torch.any(t_sum2.isnan()):
#         torch.set_printoptions(profile="full")
#         print('t_sum2')
#         print(t_sum2)
        
#     if torch.any(diff_sum2.isnan()):
#         torch.set_printoptions(profile="full")
#         print('diff_sum2')
#         print(diff_sum2)
        
#     if torch.any(ratio2.isnan()):
#         torch.set_printoptions(profile="full")
#         print('ratio2')
#         print(ratio2)
        
#         print('t_sum2')
#         print(t_sum2)

#     if torch.any(sum2.isnan()):
#         torch.set_printoptions(profile="full")
#         print('sum2')
#         print(sum2)

#     if torch.isnan(loss):
#         print('loss')
#         exit()

    if losspernode==True:
        return ratio2
                
    return loss
