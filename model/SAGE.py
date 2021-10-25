from dgllife.model.gnn import GraphSAGE
from dgllife.model.readout import MLPNodeReadout
from dgllife.model.readout import WeightedSumAndMax
from torch import nn
import torch
from torch.nn.modules.activation import Sigmoid

class SAGE(nn.Module):
    def __init__(self, node_feature_size,target_size, hidden_size=[128,128], activation = [torch.sigmoid,torch.sigmoid],aggregator_type=['lstm', 'lstm'], mid_layers = 128, task = 'classification'):
        super(SAGE, self).__init__()
        self.gnn = nn.ModuleList([GraphSAGE(in_feats=node_feature_size, hidden_feats=hidden_size, activation=activation, aggregator_type=aggregator_type)])
        self.readout = nn.ModuleList([WeightedSumAndMax(in_feats=hidden_size[-1])])
        #self.readout = nn.ModuleList([MLPNodeReadout(node_feats=hidden_size[-1], hidden_feats=mid_layers, graph_feats=mid_layers, activation=torch.sigmoid)])
        self.task_layer = None
        self.task = task
        self.task_layer = nn.ModuleList([nn.Linear(mid_layers*2, mid_layers), nn.Linear(mid_layers, target_size)])
        #self.task_layer = nn.ModuleList([nn.Linear(mid_layers, target_size)])
            

    def forward(self, g, n, e):
        node_feature = self.gnn[0](g, n)
        graph_embedding = self.readout[0](g, node_feature)
        y = self.task_layer[0](graph_embedding)
        y = torch.sigmoid(y)
        y = self.task_layer[1](y)
        if self.task == 'classification':
            y = torch.sigmoid(y)
        #print(y)
        return y

