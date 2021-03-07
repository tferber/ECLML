import torch

def energy_weighting(e, weight = 'sqrt'):
    e_in = e
    
    if weight == 'sqrt':
        e = torch.sqrt(torch.abs(e) + torch.finfo(torch.float32).eps)
    elif weight == 'square':
        e = torch.square(torch.abs(e) + torch.finfo(torch.float32).eps)
    else:
        e = torch.abs(e) + torch.finfo(torch.float32).eps
        
    return torch.where(e_in>0, e, torch.zeros_like(e))


def create_loss_dict(batch, pred):
    
    '''    
    t_energysum (true energy per crystal summed over all particles): B x V x Fractions
    t_sigfrac (true fractions):(B x V x Fractions
    p_sigfrac (predicted fractions): B x V x Fractions
    
    '''

    r_energy = batch.x[:,2]
    t_energy = batch.eik
    t_sigfrac = batch.y
    p_sigfrac = pred
    
    return {'r_energy' : r_energy,
            't_energy' : t_energy,
            't_sigfrac' : t_sigfrac,
            'p_sigfrac' : p_sigfrac}
