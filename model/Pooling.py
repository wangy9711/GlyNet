import torch as th
import torch.nn as nn
import numpy as np

from dgl.backend import pytorch as F
from dgl.base import dgl_warning
from dgl.readout import sum_nodes, mean_nodes, max_nodes, broadcast_nodes, softmax_nodes, topk_nodes

class SumPool(nn.Module):
    r"""

    Description
    -----------
    Apply sum Pool over the nodes in a graph .

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    """
    def __init__(self):
        super(SumPool, self).__init__()

    def forward(self, graph, feat):
        r"""

        Compute sum Pool.

        Parameters
        ----------
        graph : DGLGraph
            a DGLGraph or a batch of DGLGraphs
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the number
            of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to the
            batch size of input graphs.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = sum_nodes(graph, 'h')
            return readout


class AvgPool(nn.Module):
    r"""

    Description
    -----------
    Apply average Pool over the nodes in a graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    
    """
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, graph, feat):
        r"""

        Compute average Pool.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the number
            of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where
            :math:`B` refers to the batch size of input graphs.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = mean_nodes(graph, 'h')
            return readout


class MaxPool(nn.Module):
    r"""

    Description
    -----------
    Apply max Pool over the nodes in a graph.

    .. math::
        r^{(i)} = \max_{k=1}^{N_i}\left( x^{(i)}_k \right)

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.


    """
    def __init__(self):
        super(MaxPool, self).__init__()

    def forward(self, graph, feat):
        r"""Compute max Pool.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)`, where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = max_nodes(graph, 'h')
            return readout


class SortPool(nn.Module):
    r"""

    Description
    -----------
   
    Parameters
    ----------
    k : int
        The number of nodes to hold for each graph.

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

   
    """
    def __init__(self, k):
        super(SortPool, self).__init__()
        self.k = k

    def forward(self, graph, feat):
        r"""

        Compute sort Pool.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, k * D)`, where :math:`B` refers
            to the batch size of input graphs.
        """
        with graph.local_scope():
            # Sort the feature of each node in ascending order.
            feat, _ = feat.sort(dim=-1)
            graph.ndata['h'] = feat
            # Sort nodes according to their last features.
            ret = topk_nodes(graph, 'h', self.k, sortby=-1)[0].view(
                -1, self.k * feat.shape[-1])
            return ret


class GlobalAttentionPool(nn.Module):
    r"""

    Description
    -----------
    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Parameters
    ----------
    gate_nn : torch.nn.Module
        A neural network that computes attention scores for each feature.
    feat_nn : torch.nn.Module, optional
        A neural network applied to each feature before combining them with attention
        scores.

    -----
    """
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPool, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, graph, feat):
        r"""

        Compute global attention Pool.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)` where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers
            to the batch size.
        """
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            return readout
