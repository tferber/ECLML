import torch

def create_metric_dict(batch):
    
    '''
    r_baseline: baseline energy sum
    '''
    
    r_baseline = batch.eb    
    
    return {'r_baseline' : r_baseline}
