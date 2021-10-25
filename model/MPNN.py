from dgllife.model.gnn import MPNNGNN
from dgllife.model.readout import WeightedSumAndMax
from torch import nn
import torch

class MPNN(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, message_size, layers, target_size, mid_layers = 128, task = 'classification'):
        super(MPNN, self).__init__()
        self.gnn = nn.ModuleList([MPNNGNN(node_in_feats=node_feature_size, edge_in_feats=edge_feature_size, node_out_feats=message_size, edge_hidden_feats=message_size, num_step_message_passing=layers)])
        self.readout = nn.ModuleList([WeightedSumAndMax(in_feats=message_size)])
        self.task_layer = None
        self.task = task
        self.task_layer = nn.ModuleList([nn.Linear(message_size*2, mid_layers), nn.Linear(mid_layers, target_size)])
            

    def forward(self, g, n, e):
        node_feature = self.gnn[0](g, n, e)
        graph_embedding = self.readout[0](g, node_feature)
        y = self.task_layer[0](graph_embedding)
        y = self.task_layer[1](y)
        if self.task == 'classification':
            y = torch.sigmoid(y)
        return y

