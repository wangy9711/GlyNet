from dgl.nn import pytorch
from dgllife.model.readout import MLPNodeReadout
from torch import nn, tensor
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

import dgl
from dgl import function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch.nn import init
import torch
import os
import numpy as np
import random
from torch import optim
from torch import nn
from torch.nn import functional as F

def get_ori_message(edges):
    # (E, 3)
    #(hi,hj,eij,||xi-xj||2)
    xj = edges.src['x_feature']
    xi = edges.dst['x_feature']
    dist =  F.pairwise_distance(xi, xj)
    dist = torch.pow(dist,2)
    # (E,1)
    dist = dist.unsqueeze(-1)

    ret = edges.src['feature']
    ret = torch.cat((ret, edges.dst['feature']), dim=-1)
    ret = torch.cat((ret, edges.data['feature']), dim=-1)
    ret = torch.cat((ret, dist), dim=-1)

    return {'ori_m': ret}


def get_weight_vector(edges):
    xj = edges.src['x_feature']
    xi = edges.dst['x_feature']
    # (E, 3)
    v = xj-xi
    # (E, 1)
    weight = edges.data['m_x']
    # (E, 3)
    ret = v * weight
    return {'vec':ret}

class EGNN(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, message_size, x_size, layers, target_size, task = 'classification'):
        super(EGNN, self).__init__()
        self.gnn = nn.ModuleList([EGNNGNN(node_in_feats=node_feature_size, edge_in_feats=edge_feature_size, node_out_feats=message_size, x_feats=x_size, num_step_message_passing=layers)])
        self.readout = nn.ModuleList([MLPNodeReadout(node_feats=message_size, hidden_feats=message_size, graph_feats=message_size, activation=torch.relu)])
        self.task = task
        #self.task_layer = nn.ModuleList([nn.Linear(message_size, target_size), nn.Linear(message_size, target_size)])
        self.task_layer = nn.ModuleList([nn.Linear(message_size, target_size)])

    def forward(self, g, n, e, x):
        node_feature = self.gnn[0](g, n, e, x)
        graph_embedding = self.readout[0](g, node_feature)
        y = self.task_layer[0](graph_embedding)
        #y = torch.sigmoid(y)
        #y = self.task_layer[1](y)
        #if self.task == 'classification':
           # y = torch.sigmoid(y)
        return y


class EGNNGNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=64, x_feats=3, num_step_message_passing=6):
        
        super(EGNNGNN, self).__init__()
        # node's features
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        # layers
        self.num_step_message_passing = num_step_message_passing

        # Conv
        # GRU
        self.gnn_layer = ENConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_feats=edge_in_feats,
            x_feats = x_feats
        )
        self.node_bias = nn.Parameter(torch.Tensor(node_out_feats))
        self.node_mlp = nn.Sequential(
            nn.Linear(2*node_out_feats, 2*node_out_feats),
            nn.ReLU(),
            nn.Linear(2*node_out_feats, node_out_feats)
        )

    def reset_parameters(self):
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()
        nn.init.zeros_(self.node_bias)


    def forward(self, g, node_feats, edge_feats, x):
        """
        g:graph
        node_feature:(N, node_in_feature)
        edge_feature:(E, edge_feat)
        x:(N,3)
        """
        # (N, node_out_feats)
        node_feats = self.project_node_feats(node_feats) 

        for _ in range(self.num_step_message_passing):
            # M:(N, node_out_feature)
            # delta_x:(N, 3)
            M, delta_x = self.gnn_layer(g, node_feats, edge_feats, x)
            # (N, node_out_feature*2)
            h = torch.cat((node_feats, M), dim=-1)
            # (N, node_out_feature)
            node_feats = self.node_mlp(h)+self.node_bias
            # (N*3)
            x = x+delta_x

        return node_feats


class ENConv(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats, x_feats):
        super(ENConv, self).__init__()
        self._in_src_feats = in_feats
        self._out_feats = out_feats
        self._edge_feats = edge_feats
        self._x_feats = x_feats
        
        # m-mlp
        self.m_mlp = nn.Sequential(
            nn.Linear(self._in_src_feats*2+self._edge_feats+1, 2*self._out_feats),
            nn.ReLU(),
            nn.Linear(2*self._out_feats, self._out_feats)
        )
        # x_delta-mlp
        self.x_mlp = nn.Sequential(
            nn.Linear(self._out_feats, self._out_feats),
            nn.ReLU(),
            nn.Linear(self._out_feats, 1)
        )
        # mlp-bias
        self.m_bias = nn.Parameter(torch.Tensor(out_feats))
        self.x_bias = nn.Parameter(torch.Tensor(x_feats))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.m_bias)
        nn.init.zeros_(self.x_bias)
        self.m_mlp[0].reset_parameters()
        self.m_mlp[2].reset_parameters()
        self.x_mlp[0].reset_parameters()
        self.x_mlp[2].reset_parameters()

    def forward(self, graph, feat, efeat, x):
        r"""
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            The input feature of shape :(N, D_in)
        efeat : torch.Tensor
            The edge feature of shape : (E, edge_feats)
        x: torch.Tensor
            The coordinate feature of shape : (N, 3)
        Returns
        -------
        M , x_delta
        M: torch.Tensor
            Message of shape :(N, D_out)
        x_delta: torch.Tensor
            Message of shape :(N, 3)
        """
        with graph.local_scope():
            graph.edata['feature'] = efeat
            graph.ndata['feature'] = feat
            graph.ndata['x_feature'] = x
            # edata['ori_m'](E, 2*in_Feat+edge_feat+1)
            graph.apply_edges(get_ori_message)
            # 
            # edata['m']:[E, out_feat]
            graph.edata['m'] = self.m_mlp(graph.edata['ori_m'])+self.m_bias
            
            # return M:(N, out_feat)
            graph.update_all(fn.copy_e('m', 'temp'), fn.sum('temp', 'M'))
            M = graph.ndata['M']

            # x_dalta
            # nonliner
            # (E, 1)
            graph.edata['m_x'] = self.x_mlp(graph.edata['m'])+self.x_bias
            # edata['vec']
            # x_dalta:(N, 3)
            graph.updata_all(get_weight_vector, fn.sum('vec', 'x_dalta'))
            x_delta = graph.ndata['x_dalta']

            return M, x_delta


if __name__ == '__main__':
    g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
    g.ndata['x'] = torch.ones(5, 2)
    g.edata['z'] = torch.ones(4,3)
    g.update_all(fn.copy_e('z', 'y'), fn.sum('y', 'h'))
    #g.apply_edges(edge_udf)
    #print(g.edata['y'])
    print(g.ndata['h'])

