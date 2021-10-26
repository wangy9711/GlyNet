import dgl
import torch
import dgl.function as fn
import torch.nn as nn

import torch.nn.functional as F

from .Pooling import SumPool
from .mydgllib import OGBlayer

import torch
import os
import numpy as np
import random
from torch import optim
from torch import nn
from torch.nn import functional as F


class GINLayer(nn.Module):
    """
    Parameters:
    node_feats : int
        Number of input and output node features.
    in_edge_feats : int
        Number of input edge features.
    """
    def __init__(self, node_feats, in_edge_feats):
        super(GINLayer, self).__init__()

        self.in_edge_feats = nn.Linear(in_edge_feats, node_feats)
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.project_out = nn.Sequential(
            nn.Linear(node_feats, 2 * node_feats),
            nn.BatchNorm1d(2 * node_feats),
            nn.ReLU(),
            nn.Linear(2 * node_feats, node_feats)
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.in_edge_feats.reset_parameters()
        device = self.eps.device
        self.eps = nn.Parameter(torch.Tensor([0]).to(device))
        for layer in self.project_out:
            if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """
        Parameters:
        g : DGLGraph
        node_feats : FloatTensor of shape (N, node_feats)
        edge_feats : FloatTensor of shape (E, in_edge_feats)

        Returns:
        FloatTensor of shape (N, node_feats)
        """
        g = g.local_var()
        edge_feats = self.in_edge_feats(edge_feats)

        g.ndata['feat'] = node_feats
        g.apply_edges(fn.copy_u('feat', 'e'))
        g.edata['e'] = F.relu(edge_feats + g.edata['e'])
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'feat'))

        rst = g.ndata['feat']
        rst = self.project_out(rst + (1 + self.eps) * node_feats)

        return rst

class GNNOGB(nn.Module):
    """Parameters:
    in_edge_feats : int
    num_node_types : int
    hidden_feats : int
    n_layers : int
    batchnorm : bool
    activation : callable or None
    dropout : float
    gnn_type : str
    virtual_node : bool
    residual : bool
    jk : bool
    """
    def __init__(self,
                 in_edge_feats,
                 num_node_types=1,
                 hidden_feats=300,
                 n_layers=5,
                 batchnorm=True,
                 activation=F.relu,
                 dropout=0.,
                 gnn_type='gcn',
                 virtual_node=True,
                 residual=False,
                 jk=False):
        super(GNNOGB, self).__init__()

        assert gnn_type in ['gcn', 'gin'], \
            "Expect gnn_type to be either 'gcn' or 'gin', got {}".format(gnn_type)

        self.n_layers = n_layers
        # Initial node embeddings
        self.node_encoder = nn.Embedding(num_node_types, hidden_feats)
        # Hidden layers
        self.layers = nn.ModuleList()
        self.gnn_type = gnn_type
        for _ in range(n_layers):
            if gnn_type == 'gcn':
                self.layers.append(OGBlayer(in_node_feats=hidden_feats,
                                               in_edge_feats=in_edge_feats,
                                               out_feats=hidden_feats))
            else:
                self.layers.append(GINLayer(node_feats=hidden_feats,
                                               in_edge_feats=in_edge_feats))

        self.virtual_node = virtual_node
        if virtual_node:
            self.virtual_node_emb = nn.Embedding(1, hidden_feats)
            self.mlp_virtual_project = nn.ModuleList()
            for _ in range(n_layers - 1):
                self.mlp_virtual_project.append(nn.Sequential(
                    nn.Linear(hidden_feats, 2 * hidden_feats),
                    nn.BatchNorm1d(2 * hidden_feats),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_feats, hidden_feats),
                    nn.BatchNorm1d(hidden_feats),
                    nn.ReLU()))
            self.virtual_readout = SumPool()

        if batchnorm:
            self.batchnorms = nn.ModuleList()
            for _ in range(n_layers):
                self.batchnorms.append(nn.BatchNorm1d(hidden_feats))
        else:
            self.batchnorms = None

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.jk = jk

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.node_encoder.reset_parameters()
        for gnn_layer in self.layers:
            gnn_layer.reset_parameters()

        if self.virtual_node:
            nn.init.constant_(self.virtual_node_emb.weight.data, 0)
            for mlp_layer in self.mlp_virtual_project:
                for layer in mlp_layer:
                    if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                        layer.reset_parameters()

        if self.batchnorms is not None:
            for norm_layer in self.batchnorms:
                norm_layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : LongTensor of shape (N, 1)
            Input categorical node features. N for the number of nodes.
        edge_feats : FloatTensor of shape (E, in_edge_feats)
            Input edge features. E for the number of edges.

        Returns
        -------
        FloatTensor of shape (N, hidden_feats)
            Output node representations
        """
        if self.gnn_type == 'gcn':
            degs = (g.in_degrees().float() + 1).to(node_feats.device)
            norm = torch.pow(degs, -0.5).unsqueeze(-1)                # (N, 1)
            g.ndata['norm'] = norm
            g.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))
            norm = g.edata.pop('norm')

        if self.virtual_node:
            virtual_node_feats = self.virtual_node_emb(
                torch.zeros(g.batch_size).to(node_feats.dtype).to(node_feats.device))
        h_list = [self.node_encoder(node_feats)]

        for l in range(len(self.layers)):
            if self.virtual_node:
                virtual_feats_broadcast = dgl.broadcast_nodes(g, virtual_node_feats)
                h_list[l] = h_list[l] + virtual_feats_broadcast

            if self.gnn_type == 'gcn':
                h = self.layers[l](g, h_list[l], edge_feats, degs, norm)
            else:
                h = self.layers[l](g, h_list[l], edge_feats)

            if self.batchnorms is not None:
                h = self.batchnorms[l](h)

            if self.activation is not None and l != self.n_layers - 1:
                h = self.activation(h)
            h = self.dropout(h)
            h_list.append(h)

            if l < self.n_layers - 1 and self.virtual_node:
                ### Update virtual node representation from real node representations
                virtual_node_feats_tmp = self.virtual_readout(g, h_list[l]) + virtual_node_feats
                if self.residual:
                    virtual_node_feats = virtual_node_feats + self.dropout(
                        self.mlp_virtual_project[l](virtual_node_feats_tmp))
                else:
                    virtual_node_feats = self.dropout(
                        self.mlp_virtual_project[l](virtual_node_feats_tmp))

        if self.jk:
            return torch.stack(h_list, dim=0).sum(0)
        else:
            return h_list[-1]
