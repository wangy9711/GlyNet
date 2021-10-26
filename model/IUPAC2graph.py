from functools import reduce
import numpy as np
from .Config import DataConfig
import dgl
from dgl import DGLGraph
import torch
import torch
import os
import numpy as np
import random
from torch import optim
from torch import nn
from torch.nn import functional as F

cfg = DataConfig()


def str_parse(encode):
    i = 0
    j = 0
    rets = []
    L = len(encode)
    while(i<L):
        if encode[i]=='[':
            rets.append(encode[i])
            i+=1
        elif encode[i]==']':
            rets.append(encode[i])
            i+=1
        elif encode[i] == '(':
            j = i
            while(j<L):
                if encode[j] == ')':
                    break
                j+=1
            
            rets.append(encode[i:j+1])
            i = j + 1
        else:
            j = i
            while(j<L):
                if encode[j]=='(':
                    break
                j+=1
            
            rets.append(encode[i:j])
            i = j
    
    return rets

def get_link_feature(link, link_dict, link_amount):
    if cfg.use_link_inof:
        ret = torch.zeros((1, link_amount))
        id = link_dict.get(link, link_dict["Unknown"])
        ret[0][id] = 1
    else:
        ret = torch.zeros((1, 1))
        ret[0][0] = 1
    return ret


def get_gly_feature(gly, gly_dict, gly_amount, monose_feature):
    use_reduce = 0
    if cfg.use_reduce_info:
        use_reduce = 1
    use_mo = 0
    if cfg.use_monose_info:
        use_mo = 6
    
    ret = torch.zeros((1, 1))

    this_gly = ""
    if gly in monose_feature.keys():
        this_gly = gly
    else:
        this_gly = "Unknown"

    for i in range(use_mo):
        ret[0][use_reduce+i] = monose_feature[this_gly][i]
    
    gly_index = gly_dict.get(gly, gly_dict["Unknown"])

    ret[0][0] = gly_index
    return ret

def IUPAC2graph(
    IUPAC, 
    gly_dict, 
    gly_amount, 
    link_dict, 
    link_amount,
    monose_feature):
    all_element = str_parse(IUPAC)
    all_element.reverse()
    L = len(all_element)
    i = 0
    g = DGLGraph()
    g.add_nodes(1)
    curr = 0
    node_stack = []
    edge_count = 0
    while(i<L):
        if all_element[i][0] == '(':
            this_link = all_element[i][1:-1]
            g.add_nodes(1)
            g.add_edge(curr, g.num_nodes()-1)
            g.edges[[edge_count]].data['x'] = get_link_feature(
                this_link, 
                link_dict, 
                link_amount)
            edge_count += 1
            g.add_edge(g.num_nodes()-1,curr)
            g.edges[[edge_count]].data['x'] = get_link_feature(
                this_link, 
                link_dict, 
                link_amount)
            edge_count += 1

            curr = g.num_nodes()-1
        
        elif all_element[i] == '[':
            curr = node_stack.pop()
        
        elif all_element[i] == ']':
            node_stack.append(curr)

        else:
            g.nodes[[curr]].data['x'] = get_gly_feature(
                all_element[i], 
                gly_dict, 
                gly_amount, 
                monose_feature)
        
        i+=1
    
    if cfg.use_reduce_info:
        g.nodes[[0]].data['x'][0] = 1
    #转为无向图

    #g = dgl.to_bidirected(g)
    return g

def get_dict(all_gly):
    # 0表示无
    rets = {}
    begin = 0
    for item in all_gly:
        rets[item] = begin
        begin +=1
    return rets

def get_data_from_code(
    code, 
    given_gly_list, 
    given_link_list, 
    monose_feature):
    gly_dict = get_dict(given_gly_list)
    link_dict = get_dict(given_link_list)
    gly_amount = len(given_gly_list)
    link_amount = len(given_link_list)
    g = IUPAC2graph(
        code, 
        gly_dict, 
        gly_amount, 
        link_dict, 
        link_amount, 
        monose_feature)
    
    return g
                


if __name__ == '__main__':
    code = "Fuc(a1-3)[Gal(b1-4)]GlcNAc(b1-2)Man(a1-3)[Fuc(a1-3)[Gal(b1-4)]GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"
    g = get_data_from_code(code, cfg.glycan, cfg.link, cfg.monose)
    print(g.ndata['feature'].shape)
    

