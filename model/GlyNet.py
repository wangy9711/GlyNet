from dgllife.model.gnn import GNNOGB
from dgllife.model.readout import MLPNodeReadout
from dgllife.model.readout import WeightedSumAndMax
from torch import nn
import torch
from torch.nn.modules.activation import Sigmoid

class GlyNet(nn.Module):
    def __init__(self, edge_feature_size, node_type, target_size, message_size, graph_size, mid_size, layers, dropout, residual, task = 'classification'):
        super(GlyNet, self).__init__()
        self.edge_feature_size = edge_feature_size
        self.node_type = node_type
        self.target_size = target_size
        self.message_size = message_size
        self.graph_size = graph_size
        self.mid_size = mid_size
        self.layers = layers
        self.task = task
        self.dropout = dropout
        self.residual = residual

        self.gnn = nn.ModuleList([GNNOGB(in_edge_feats=self.edge_feature_size, num_node_types=self.node_type, hidden_feats=self.message_size, gnn_type='gin', dropout=self.dropout,residual=self.residual, n_layers=self.layers)])

        #self.readout = nn.ModuleList([WeightedSumAndMax(in_feats=hidden_size[-1])])
        self.readout = nn.ModuleList([MLPNodeReadout(node_feats=self.message_size, hidden_feats=self.graph_size, graph_feats=self.graph_size, activation=torch.sigmoid)])

        self.task_layer = nn.ModuleList([nn.Linear(self.graph_size, self.mid_size), nn.Linear(self.mid_size, target_size)])
        #self.task_layer = nn.ModuleList([nn.Linear(mid_layers, target_size)])
            
    def forward(self, g, n, e):
        node_feature = self.gnn[0](g, n, e)
        graph_embedding = self.readout[0](g, node_feature)
        y = self.task_layer[0](graph_embedding)
        y = torch.sigmoid(y)
        y = self.task_layer[1](y)
        if self.task == 'classification':
            y = torch.sigmoid(y)
        #print(y)
        return y

