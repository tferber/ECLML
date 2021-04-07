import torch
from torch_geometric.utils import to_dense_batch 

def core_weighting(theta, phi, pred, e, batch, alpha=3, beta=0.05):
        
    # get the currently predicted maxima per graph:
    
    # convert sparse to dense
    # x_dense, mask = to_dense_batch(x, batch)
    # x = x_dense.view(-1, x.size(-1))[mask.view(-1)]
    ew_batch, ew_batch_mask = to_dense_batch(e.unsqueeze(dim=-1)*pred, batch)
    theta_batch, theta_batch_mask = to_dense_batch(theta, batch)
    phi_batch, phi_batch_mask = to_dense_batch(phi, batch)
    
    # get the maximum per graph
    idx_ewmax = torch.argmax(ew_batch, dim=1)#, keepdim=True)

    # select the corresponding theta and phi values
    thetamax = torch.gather(theta_batch, -1, idx_ewmax)[batch]
    phimax = torch.gather(phi_batch, -1, idx_ewmax)[batch]
    
    # calculate the distance    
    d0 = torch.sqrt((theta-thetamax[:,0])**2+(phi-phimax[:,0])**2)
    d1 = torch.sqrt((theta-thetamax[:,1])**2+(phi-phimax[:,1])**2)
    cw = torch.stack([torch.sqrt((theta-thetamax[:,0])**2+(phi-phimax[:,0])**2),
                     torch.sqrt((theta-thetamax[:,1])**2+(phi-phimax[:,1])**2)], dim=1)
    
    cw = torch.exp((cw**alpha)/beta)
    cw = torch.cat((cw, torch.ones_like(theta).unsqueeze(dim=-1)), dim=1)

    return cw

def energy_weighting(e, usesqrt = True):
#     e_in = e
    
    if usesqrt:
        e = torch.sqrt(torch.abs(e) + torch.finfo(torch.float32).eps)
    else:
        e = torch.abs(e) + torch.finfo(torch.float32).eps
        
#     return torch.where(e_in>0, e, torch.zeros_like(e)) #this makes a problem is e_in is 0, it bypasses .eps
    return e


def create_loss_dict(batch, pred, sort=True):
    
    '''    
    r_theta (rec theta per node): (B x V)
    r_phi (rec theta per node): (B x V)
    r_energy (rec theta per node): (B x V)
    
    t_energy (true energy):(B x V) x Fractions
    t_sigfrac (true fractions):(B x V) x Fractions
    p_sigfrac (predicted fractions): B x V) x Fractions
    
    '''

    r_theta = batch.x[:,0] # reconstructed crystal theta
    r_phi = batch.x[:,1] # reconstructed crystal phi
    r_energy = batch.x[:,2] # reconstructed crystal energy
                   
    t_energy = batch.eik # true energy
    t_sigfrac = batch.y #true energy fraction
    p_sigfrac = pred #predicted energy fraction
    
    return {'r_theta' : r_theta,
            'r_phi' : r_phi,
            'r_energy' : r_energy,
            't_energy' : t_energy,
            't_sigfrac' : t_sigfrac,
            'p_sigfrac' : p_sigfrac}

