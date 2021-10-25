import numpy as np
import torch
import os
from torch.autograd import Variable
import dgl


from .Config import TrainConfig

train_cfg = TrainConfig() 

def read_file(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
        data = data.split("\n")
        all_data = []
        for item in data:
            if item != "":
                all_data.append(item)
        
        return all_data

def collatef(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def get_lr_step(lr, lr_decay, schedule, epochs):
    return (lr-lr*lr_decay)/(epochs*schedule[1]-epochs*schedule[0])

def evaluation(output, target, use_cuda):
    pred = torch.tensor([1 if num >= 0.5 else 0 for num in output])
    if use_cuda!='cpu':
        pred = pred.to(use_cuda)

    L = target.shape[0]
    count = 0
    for i in range(L):
        if pred[i]==target[i]:
            count+=1

    ACC = count/torch.tensor(float(L))
    return ACC.item()

def stat(output, target, use_cuda):
    pred = torch.tensor([1 if num >= 0.5 else 0 for num in output])
    if use_cuda!='cpu':
        pred = pred.to(use_cuda)
    L = target.shape[0]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(L):
        if pred[i]==1 and target[i]==1 :
            TP += 1
        elif pred[i]==1 and target[i]==0:
            FP += 1
        elif pred[i]==0 and target[i]==1:
            FN += 1
        elif pred[i]==0 and target[i]==0:
            TN += 1
    assert((TP+TN+FP+FN)==L)
    return [TP,TN,FP,FN]

def save_model(model, ACC, path, filename):
    if not os.path.isdir(path):
        os.mkdir(path)
    
    file = path + filename
    torch.save({'state_dict':model.state_dict(), 'ACC':ACC}, file)

def train_model(train_loader, model, use_cuda, lossf, opt, epoch, logger, logtrain):
    model.train()
    for g, target in train_loader:
        
        if use_cuda!='cpu':
            g = g.to(use_cuda)
            target = target.to(use_cuda)
        n = g.ndata['x'].long().squeeze(dim=1)
        e = g.edata['x']
        n, e, target = Variable(n), Variable(e), Variable(target)
        opt.zero_grad()
        output = model(g, n, e)
        
        loss = lossf(output, target)
        ACC = evaluation(output, target, use_cuda)
        L = target.shape[0]
        logger.update_stat_value('train_loss', loss.item(),L)
        logger.update_stat_value('train_ACC', ACC, L)
        loss.backward()
        opt.step()
    
    loss_value = logger.get_stat_value('train_loss')
    ACC_value = logger.get_stat_value('train_ACC')
    logger.write_value(logtrain, [ACC_value, loss_value])

    print('Epoch: [{0}/{1}] Train Avg Loss {loss_value:.3f}; Train Avg ACC {ACC_value:.3f};'.format(epoch, train_cfg.epochs, loss_value=loss_value, ACC_value=ACC_value))
    logger.clear_stat_log()
    return ACC_value


def valid_model(valid_loader, model, use_cuda, lossf, epoch, logger, logvalid):
    model.eval()
    for g, target in valid_loader:
        if use_cuda!='cpu':
            g = g.to(use_cuda)
            target = target.to(use_cuda)
        n = g.ndata['x'].long().squeeze(dim=1)
        e = g.edata['x']
        n, e, target = Variable(n), Variable(e), Variable(target)
        output = model(g, n, e)
        loss = lossf(output, target)
        score = stat(output, target, use_cuda)
        L = target.shape[0]
        logger.update_stat_value('test_loss', loss.item(), L)
        logger.update_value('test_TP', score[0])
        logger.update_value('test_TN', score[1])
        logger.update_value('test_FP', score[2])
        logger.update_value('test_FN', score[3])
    
    loss_value = logger.get_stat_value('test_loss')
    TP = logger.get_value('test_TP')
    TN = logger.get_value('test_TN')
    FP = logger.get_value('test_FP')
    FN = logger.get_value('test_FN')
    pre = TP/(TP+FP)
    recall = TP/(TP+FN)
    acc = (TP+TN)/(TP+TN+FP+FN)
    Fscore = (2*pre*recall)/(pre+recall)
    logger.write_value(logvalid, [loss_value, pre, recall, acc, Fscore])
    print('Epoch: [{0}/{1}] Valid Avg Loss {loss_value:.3f}; Valid Avg ACC {ACC_value:.3f};'.format(epoch, train_cfg.epochs, loss_value=loss_value, ACC_value=acc))
    logger.clear_stat_log()

    return acc, loss_value

def test_model(test_loader, model, lossf, use_cuda, logger, detail = None):
    model.eval()
    for g, target in test_loader:
        if use_cuda!='cpu':
            g = g.to(use_cuda)
            target = target.to(use_cuda)
        n = g.ndata['x'].long().squeeze(dim=1)
        e = g.edata['x']
        n, e, target = Variable(n), Variable(e), Variable(target)
        
        output = model(g, n, e)
        loss = lossf(output, target)
        score = stat(output, target, use_cuda)
        L = target.shape[0]
        if detail:
            for i in range(L):
                detail.write(str(output[i][0].item())+','+str(target[i][0].item())+'\n')
    
        logger.update_stat_value('test_loss', loss.item(), L)
        logger.update_value('test_TP', score[0])
        logger.update_value('test_TN', score[1])
        logger.update_value('test_FP', score[2])
        logger.update_value('test_FN', score[3])
    
    loss_value = logger.get_stat_value('test_loss')
    TP = logger.get_value('test_TP')
    TN = logger.get_value('test_TN')
    FP = logger.get_value('test_FP')
    FN = logger.get_value('test_FN')
    pre = TP/(TP+FP)
    recall = TP/(TP+FN)
    acc = (TP+TN)/(TP+TN+FP+FN)
    Fscore = (2*pre*recall)/(pre+recall)
    logger.clear_stat_log()

    return [loss_value, pre, recall, acc, Fscore]


def write_ans(inputdata, score, file_name):
    f = open(file_name, 'w')
    for code, s in zip(inputdata, score):
        imm = 1 if s>0.5 else 0
        f.write(code+','+str(s)+','+str(imm)+'\n')
    
    f.close()

