import os
import shutil
import torch
import os
import numpy as np
import random
from torch import optim
from torch import nn
from torch.nn import functional as F

from tensorboard_logger import configure, log_value

class Logger(object):
    def __init__(self, log_path,log_list):
        self.log_path = log_path
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        self.stat_list = log_list
        self.stat_dict = {}
        for item in self.stat_list:
            self.stat_dict[item] = [0,0]

    def update_stat_value(self, item, value, count):
        self.stat_dict[item][0] += value * count
        self.stat_dict[item][1] += count
    
    def update_value(self, item, value):
        self.stat_dict[item][0] += value
    
    def get_stat_value(self, item):
        if self.stat_dict[item][1] == 0:
            return 0
        return self.stat_dict[item][0]/self.stat_dict[item][1]
    
    def get_value(self, item):
        return self.stat_dict[item][0]
    
    def clear_stat_log(self):
        for item in self.stat_dict.keys():
            self.stat_dict[item] = [0,0]
    
    @staticmethod
    def write_value(tfile, value):
        for i in range(len(value)-1):
            tfile.write(str(value[i])+',')
        tfile.write(str(value[-1])+'\n')