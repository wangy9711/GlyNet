import torch.utils.data as data
import torch
import os
import numpy as np
import random
from torch import optim
from torch import nn
from torch.nn import functional as F
from .IUPAC2graph import get_data_from_code
from .Config import DataConfig

cfg = DataConfig()

class GlyBase(data.Dataset):
    def __init__(self, raw_data, test=False):
        self.raw_data = raw_data
        self.gly_list = cfg.glycan
        self.link_list = cfg.link
        self.monose_feature = cfg.monose
        self.test_falg = test

    def __getitem__(self, index):
        if not self.test_falg:
            return self.get_data_with_label(index)
        else:
            return self.get_data_without_label(index)
    
    def get_data_with_label(self, index):
        raw_str = self.raw_data[index]
        if '"' in raw_str:
            a = raw_str.split('"')
            code = a[1]
            target = int(a[2].split(',')[1])
            
        else:
            a = raw_str.split(',')
            code = a[0]
            target = int(a[1])

        g = get_data_from_code(
            code, 
            self.gly_list, 
            self.link_list, 
            self.monose_feature)
        return g, [float(target)]
    
    def get_data_without_label(self, index):
        code = self.raw_data[index]
        g = get_data_from_code(
            code, 
            self.gly_list, 
            self.link_list, 
            self.monose_feature)
        return g

    def __len__(self):
        return len(self.raw_data)