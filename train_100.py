import torch
import os
import numpy as np
import random
from torch import optim
from torch import nn
from torch.nn import functional as F

from model.GlyNet import GlyNet
from model.Config import DataConfig, TrainConfig, ModelConfig
from model.Utils import read_file
from model.Utils import collatef
from model.Utils import get_lr_step
from model.Utils import valid_model
from model.Utils import train_model
from model.Utils import save_model
from model.Utils import test_model
from model.database import GlyBase
from model.Logger import Logger


model_cfg = ModelConfig()
train_cfg = TrainConfig()
data_cfg = DataConfig()

if __name__ == '__main__':
    
    logger = Logger(train_cfg.logPath, ['train_loss', 'train_ACC', 'test_loss', 'test_TP', 'test_TN', 'test_FP', 'test_FN'])
    log_test = open(train_cfg.logPath + 'test.csv', 'w')
    allacc = 0
    for t in range(train_cfg.train_times):

        log_train = open(train_cfg.logPath+str(t)+train_cfg.train_log_file, 'w')
        log_valid = open(train_cfg.logPath+str(t)+train_cfg.valid_log_file, 'w')
        log_test_detail = open(train_cfg.logPath+str(t)+'test_detils.csv', 'w')
        
        all_data = read_file(train_cfg.datasetPath)
        idx = list(range(len(all_data)))
        random.shuffle(idx)
        test_ids = [all_data[i] for i in idx[0:train_cfg.test_num]]
        valid_ids = [all_data[i] for i in idx[train_cfg.test_num:train_cfg.test_num+train_cfg.valid_num]]
        train_ids = [all_data[i] for i in idx[train_cfg.test_num+train_cfg.valid_num:]]

        data_test = GlyBase(test_ids)
        data_train = GlyBase(train_ids)
        data_valid = GlyBase(valid_ids)

        train_loader = torch.utils.data.DataLoader(
            data_train, 
            batch_size=train_cfg.batch_size, 
            shuffle=True, 
            collate_fn=collatef,
            num_workers=1,
            pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            data_valid, 
            batch_size=train_cfg.batch_size, 
            shuffle=True, 
            collate_fn=collatef, 
            num_workers=1,  
            pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            data_test, 
            batch_size=train_cfg.batch_size, 
            shuffle=True, 
            collate_fn=collatef, 
            num_workers=1,  
            pin_memory=True)

        g, target = data_train[0]
        node_size = g.ndata['x'].shape[1]
        edge_size = g.edata['x'].shape[1]
        node_type=len(data_cfg.glycan)
        message_embedding_size = model_cfg.message_embedding_size
        graph_embedding_size = model_cfg.graph_embedding_size
        mid_embedding_size = model_cfg.mid_embedding_size
        layers = model_cfg.layers
        dropout = model_cfg.dropout
        residual = model_cfg.residual

        target_size = len(target)

        model = GlyNet(
            edge_feature_size = edge_size, 
            node_type=node_type, 
            target_size=target_size, 
            message_size=message_embedding_size, 
            graph_size = graph_embedding_size,
            mid_size = mid_embedding_size,
            layers=layers,
            dropout = dropout,
            residual=residual)
            
        opt = optim.Adam(model.parameters(), lr = train_cfg.lr)
        lossf = F.binary_cross_entropy

        lr_step = get_lr_step(
            train_cfg.lr, 
            train_cfg.lr_decay, 
            train_cfg.lr_schedule, 
            train_cfg.epochs)

        if train_cfg.use_cuda and torch.cuda.is_available():
            use_cuda = 'cuda:' + train_cfg.use_cuda
        else:
            use_cuda = 'cpu'
        if use_cuda != 'cpu':
            model = model.to(use_cuda)

        lr = train_cfg.lr
        max_ACC = 0
        min_loss = 100
        for epoch in range(train_cfg.epochs):
            if epoch > train_cfg.epochs * train_cfg.lr_schedule[0] and epoch < train_cfg.epochs * train_cfg.lr_schedule[1]:
                lr = lr - lr_step
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

            train_ACC = train_model(train_loader, model, use_cuda, lossf, opt, epoch, logger, log_train)
            valid_ACC, valid_loss = valid_model(valid_loader, model, use_cuda, lossf, epoch, logger, log_valid)
            aim_ACC = int((valid_ACC+0.00001)*100)
            if min_loss>valid_loss:
            #if (aim_ACC>max_ACC and valid_loss<min_loss) or (aim_ACC==max_ACC and valid_loss<min_loss):
                min_loss = valid_loss
                max_ACC = valid_ACC 
                save_model(model, valid_ACC, train_cfg.checkpoint_path, str(t)+'checkpoint.pth')
                print('Train ACC:{0:.3f};Valid ACC:{1:.3f};Valid loss:{2:.3f}'.format(train_ACC, valid_ACC, valid_loss))
                print('Save new checkpoint!')

        checkpoint = torch.load(train_cfg.checkpoint_path + str(t)+'checkpoint.pth')
        model.load_state_dict(checkpoint['state_dict'])
        test_score = test_model(test_loader, model, lossf, use_cuda, logger, log_test_detail)
        print('[{0}]Test ACC:{1:.3f};'.format(t, test_score[3]))
        logger.write_value(log_test, test_score)
        log_train.close()
        log_valid.close()
        allacc += test_score[3]
    
    log_test.close()
    print(allacc/train_cfg.train_times)
    
    


