import torch
from torch_geometric.nn import GravNetConv, BatchNorm, global_mean_pool
import torch.nn.functional as F
    
# based on https://arxiv.org/pdf/1902.07987.pdf
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gravnet_conv.html
class GravNet(torch.nn.Module):
    def __init__(self, in_channels):
        super(GravNet, self).__init__()

        n_feature_transform = 16 #
        out_channels = 16 # 
        space_dimensions = 3 # 'S'
        propagate_dimensions = 16 # 'F_LR'
        k = 12 # k-nearest neighbours
        n_classes = 3
        batchnorm = 0.05 # 1-batchnorm in tensorflow!
        n_gravstack = 3 # number of gravnet blocks
        
        # gravnet stack: 1
        self.ft1_1 = torch.nn.Linear(in_features=in_channels*2, out_features=n_feature_transform)
        self.ft1_2 = torch.nn.Linear(in_features=n_feature_transform, out_features=n_feature_transform)
        self.ft1_3 = torch.nn.Linear(in_features=n_feature_transform, out_features=n_feature_transform)
        self.gn1 = GravNetConv(in_channels=n_feature_transform, 
                               out_channels=out_channels, 
                               space_dimensions=space_dimensions, 
                               propagate_dimensions=propagate_dimensions, 
                               k=k)
        self.bn1 = BatchNorm(out_channels, momentum=batchnorm)
        
        
        
        # gravnet stack: 2
        self.ft2_1 = torch.nn.Linear(in_features=out_channels, out_features=n_feature_transform)
        self.ft2_2 = torch.nn.Linear(in_features=n_feature_transform, out_features=n_feature_transform)
        self.ft2_3 = torch.nn.Linear(in_features=n_feature_transform, out_features=n_feature_transform)
        self.gn2 = GravNetConv(in_channels=n_feature_transform, 
                               out_channels=out_channels, 
                               space_dimensions=space_dimensions, 
                               propagate_dimensions=propagate_dimensions, 
                               k=k)
        self.bn2 = BatchNorm(out_channels, momentum=batchnorm)
        
        
        # gravnet stack: 3
        self.ft3_1 = torch.nn.Linear(in_features=out_channels, out_features=n_feature_transform)
        self.ft3_2 = torch.nn.Linear(in_features=n_feature_transform, out_features=n_feature_transform)
        self.ft3_3 = torch.nn.Linear(in_features=n_feature_transform, out_features=n_feature_transform)
        self.gn3 = GravNetConv(in_channels=n_feature_transform, 
                               out_channels=out_channels, 
                               space_dimensions=space_dimensions, 
                               propagate_dimensions=propagate_dimensions, 
                               k=k)
        self.bn3 = BatchNorm(out_channels, momentum=batchnorm)
        
        
        # final layers
        self.final1 = torch.nn.Linear(in_features=n_gravstack*out_channels, out_features=64)
        self.final2 = torch.nn.Linear(in_features=64, out_features=n_classes+1)
        self.final3 = torch.nn.Linear(in_features=n_classes+1, out_features=n_classes)

    def forward(self, data):
      
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
        # list that will hold the gravnet outputs
        feat = []
        
        # append mean of all features ("GlobalExchange" in original paper)
        # see https://github.com/rusty1s/pytorch_geometric/discussions/2123
        out = global_mean_pool(x, batch)
        x = torch.cat([x, out[batch]], dim=-1)

        # gravnet stacks
        x = F.elu(self.ft1_1(x))
        x = F.elu(self.ft1_2(x))
        x = torch.tanh(self.ft1_3(x)) #torch.tanh
        x = self.gn1(x, batch)
        x = self.bn1(x)
        feat.append(x)
        
        x = F.elu(self.ft2_1(x))
        x = F.elu(self.ft2_2(x))
        x = torch.tanh(self.ft2_3(x)) #torch.tanh
        x = self.gn2(x, batch)
        x = self.bn2(x)
        feat.append(x)
        
        x = F.elu(self.ft3_1(x))
        x = F.elu(self.ft3_2(x))
        x = torch.tanh(self.ft3_3(x))#torch.tanh
        x = self.gn3(x, batch)
        x = self.bn3(x)
        feat.append(x)
        
        # final classification
        # concatenate all outputs from the previous layers
        x = torch.cat(feat, dim=1)

        x = F.relu(self.final1(x))
        x = F.relu(self.final2(x))
        x = self.final3(x)
        out = F.softmax(x, dim=1)
        
        return out
