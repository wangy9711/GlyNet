from dgllife.model.gnn import MPNNGNN
from dgllife.model.readout import WeightedSumAndMax
from torch import nn
from torch.autograd import Variable
import torch
from Config import ModelConfig
from torch.nn import functional as F
from torch.serialization import INT_SIZE
import torch
import os
import numpy as np
import random
from torch import optim
from torch import nn

cfg = ModelConfig()

class MPNN(nn.Module):
    def __init__(
        self, 
        node_feature_size, 
        edge_feature_size, 
        message_size, 
        layers, 
        target_size, 
        mid_layers = 128, 
        task = 'classification'):
        super(MPNN, self).__init__()
        self.gnn = nn.ModuleList([MPNNGNN(
            node_in_feats=node_feature_size, 
            edge_in_feats=edge_feature_size, 
            node_out_feats=message_size, 
            edge_hidden_feats=message_size, 
            num_step_message_passing=layers)])
        self.readout = nn.ModuleList([WeightedSumAndMax(in_feats=message_size)])
        self.task_layer = None
        self.task = task
        self.task_layer = nn.ModuleList([
            nn.Linear(message_size*2, mid_layers), 
            nn.Linear(mid_layers, target_size)])
            

    def forward(self, g, n, e):
        node_feature = self.gnn[0](g, n, e)
        graph_embedding = self.readout[0](g, node_feature)
        y = self.task_layer[0](graph_embedding)
        y = self.task_layer[1](y)
        if self.task == 'classification':
            y = torch.sigmoid(y)
        return y



class MYMPNN(nn.Module):
    def __init__(
        self, 
        node_size, 
        edge_size, 
        hidden_size, 
        message_size, 
        layers, 
        target_size):
        super(MYMPNN, self).__init__()
        self.messagef = nn.ModuleList([MessageFunc(
            hidden_size,  
            message_size, 
            edge_size)])
        self.updatef = nn.ModuleList([UpdateFunc(
            message_size, 
            hidden_size)])
        self.readf = nn.ModuleList([ReadFunc(
            hidden_size,
            target_size)])
        self.layers = layers
        self.hidden_size = hidden_size
    
    def forward(self, g, h0, e):
        h = []
        h_t = torch.cat([h0, Variable(torch.zeros(h0.size(0), h0.size(1), self.hidden_size - h0.size(2)).type_as(h0.data))], 2)
        h.append(h_t)
        for i in range(self.layers):
            e_feature = e.view(-1, e.size(3))
            h_feature = h[i].view(-1, h[i].size(2))
            m = self.messagef[0].forward(h[i], h_feature, e_feature)
            m = m.view(h[0].size(0), h[0].size(1), -1, m.size(1))
            m = torch.unsqueeze(g, 3).expand_as(m) * m
            m = torch.squeeze(torch.sum(m, 1))
            h_t = self.updatef[0].forward(h[i], m)
            h_t = (torch.sum(h0, 2)[..., None].expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)

        res = self.readf[0].forward(h)
        rts = torch.sigmoid(res)
        return rts

class MessageFunc(nn.Module):
    def __init__(self,in_size, out_size, edge_size):
        super(MessageFunc, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.edge_size = edge_size
        self.net = nn.ModuleList([Net(
            self.edge_size, 
            cfg.net_layers, 
            self.in_size*self.out_size)])

    def forward(self, h_v, h_w, e_vw):
        m = self.net[0](e_vw)
        m = m.view(-1, self.out_size, self.in_size)
        h_w_rows = h_w[..., None].expand(h_w.size(0), h_w.size(1), h_v.size(1)).contiguous()
        h_w_rows = h_w_rows.view(-1, self.in_size)
        out = torch.bmm(m, torch.unsqueeze(h_w_rows, 2))
        output = torch.squeeze(out)
        return output


class UpdateFunc(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpdateFunc, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.net = nn.ModuleList([nn.GRU(in_size, out_size)])
    
    def forward(self, h, m):
        h_in = h.view(-1, h.size(2))
        m_in = m.view(-1, m.size(2))
        new_h = self.net[0](m_in[None, ...], h_in[None, ...])[0]
        return torch.squeeze(new_h).view(h.size())


class ReadFunc(nn.Module):
    def __init__(self, in_size, out_size):
        super(ReadFunc, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.layers = nn.ModuleList([
            Net(
                2*self.in_size, 
                cfg.layers, 
                self.out_size), 
            Net(
                self.in_size,
                cfg.layers, 
                self.out_size)])



    def forward(self, h):
        out = Variable(torch.Tensor(h[0].size(0), self.out_size).type_as(h[0].data).zero_())
        for i in range(h[0].size(0)):
            nn_res = (self.layers[0](torch.cat([h[0][i,:,:], h[-1][i,:,:]],1)))*self.layers[1](h[-1][i,:,:])
            nn_res = (torch.sum(h[0][i,:,:],1)[...,None].expand_as(nn_res)>0).type_as(nn_res)* nn_res
            out[i,:] = torch.sum(nn_res, 0)
        
        return out


class Net(nn.Module):
    def __init__(self, in_size, layers_size, out_size):
        super(Net, self).__init__()
        self.n_layers = len(layers_size)
        mlist = [nn.Linear(in_size, layers_size[0])]
        for i in range(self.n_layers-1):
            mlist.append(nn.Linear(layers_size[i], layers_size[i+1]))
        mlist.append(nn.Linear(layers_size[-1], out_size))
        self.layers = nn.ModuleList(mlist)
    
    def forward(self, x):
        count = 1
        size = x.size()[1:]
        for item in size:
            count = count*item
        
        x = x.contiguous().view(-1, count)
        for i in range(self.n_layers):
            x = F.relu(self.layers[i](x))
        
        return self.layers[-1](x)

        

