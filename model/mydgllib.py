import dgl
import torch
import dgl.function as fn
import torch.nn as nn

import torch.nn.functional as F

from .Pooling import SumPooling
import torch as th
from torch import nn
from torch.nn import init

from dgl import function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair



class OGBlayer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats):
        super(OGBlayer, self).__init__()

        self.in_node_feats = nn.Linear(in_node_feats, out_feats)
        self.in_edge_feats = nn.Linear(in_edge_feats, out_feats)
        self.residual = nn.Embedding(1, out_feats)

    def reset_parameters(self):
        self.in_node_feats.reset_parameters()
        self.in_edge_feats.reset_parameters()
        self.residual.reset_parameters()

    def forward(self, g, node_feats, edge_feats, degs, norm):
        """
        Parameters:
        g : DGLGraph
        node_feats : FloatTensor of shape (N, in_node_feats)
        edge_feats : FloatTensor of shape (E, in_edge_feats)
        degs : FloatTensor of shape (N, 1)
        norm : FloatTensor of shape (E, 1)
            
        Returns:
        FloatTensor of shape (N, out_feats)
        """
        g = g.local_var()
        node_feats = self.in_node_feats(node_feats)
        edge_feats = self.in_edge_feats(edge_feats)

        g.ndata['feat'] = node_feats
        g.apply_edges(fn.copy_u('feat', 'e'))
        edge_feats = F.relu(g.edata['e'] + edge_feats)
        g.edata['e'] = norm * edge_feats
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'feat'))

        residual_node_feats = node_feats + self.residual.weight
        residual_node_feats = F.relu(residual_node_feats)
        residual_node_feats = residual_node_feats * 1. / degs.view(-1, 1)

        rst = g.ndata['feat'] + residual_node_feats

        return rst

class NNConv(nn.Module):
    r"""

    Description
    -----------
    Graph Convolution layer 
    .. math::
        h_{i}^{l+1} = h_{i}^{l} + \mathrm{aggregate}\left(\left\{
        f_\Theta (e_{ij}) \cdot h_j^{l}, j\in \mathcal{N}(i) \right\}\right)

    where :math:`e_{ij}` is the edge feature, :math:`f_\Theta` is a function
    with learnable parameters.

    Parameters
    ----------
    in_feats : int
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    edge_func : callable activation function/layer
    aggregator_type : str
        Aggregator type to use (``sum``, ``mean`` or ``max``).
    residual : bool, optional
        If True, use residual connection. Default: ``False``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_func,
                 aggregator_type='mean',
                 residual=False,
                 bias=True):
        super(NNConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.edge_func = edge_func
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        """
        gain = init.calculate_gain('relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, efeat):
        """
        Compute MPNN Graph Convolution layer.

        Parameters:
        graph : DGLGraph
        feat : torch.Tensor or pair of torch.Tensor
            
        efeat : torch.Tensor

        Returns:
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # (n, d_in, 1)
            graph.srcdata['h'] = feat_src.unsqueeze(-1)
            # (n, d_in, d_out)
            graph.edata['w'] = self.edge_func(efeat).view(-1, self._in_src_feats, self._out_feats)
            # (n, d_in, d_out)
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self.reducer('m', 'neigh'))
            rst = graph.dstdata['neigh'].sum(dim=1) # (n, d_out)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst
