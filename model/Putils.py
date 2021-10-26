import os
import tqdm

import numpy as np
import dgl
import torch
from torch.autograd import Variable
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import is_aa
from Bio import SeqIO

from .Config import ModelConfig, DataConfig, TrainConfig

cfg = ModelConfig()
data_cfg = DataConfig()
train_cfg = TrainConfig()


def stat_atoms(root):
    dirs = os.listdir(root)
    atom_map = {}
    count = 0
    parser = PDBParser(PERMISSIVE = True, QUIET = True)
    for file in tqdm.tqdm(dirs):
        if '.ent' not in file:
            continue
        protein_name = file[3:7]
        data = parser.get_structure(protein_name,root+file)
        for model in data.get_list():
            for chain in model.get_list():
                for residue in chain.get_list():
                    if not is_aa(residue):
                        continue
                    tags = residue.get_full_id()
                    if tags[3][0] != " ":
                        continue
                    for atom in residue.get_list():
                        name = atom.get_name()
                        if name in atom_map.keys():
                            continue
                        atom_map[name] = count
                        count += 1
    
    print(count)
    print(atom_map)

def stat_file(file_name):
    atom_map = {}
    count = 0
    parser = PDBParser(PERMISSIVE = True, QUIET = True)
    data = parser.get_structure('3f5f',file_name)
    for model in data.get_list():
        for chain in model.get_list():
            for residue in chain.get_list():
                if not is_aa(residue):
                    continue
                tags = residue.get_full_id()
                if tags[3][0] != " ":
                    continue
                for atom in residue.get_list():
                    name = atom.get_name()
                    if name == 'P2':
                        assert(1==2)
                    if name in atom_map.keys():
                        continue
                    atom_map[name] = count
                    count += 1




def get_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


class SampleItem:
    def __init__(self, name=None, site=None, marker_peptide=None, chain = None, lable=[]):
        self.name = name
        self.site = site
        self.marker_peptide = marker_peptide
        self.chain = chain
        self.label = lable
        


def read_index(index_file, uniport_file, init=False):
    # 读取uniport文件，建立name-id索引
    uniport_dict = {}
    uniport_list = list(SeqIO.parse(uniport_file, "fasta"))
    count = 0
    for item in uniport_list:
        code = item.name.split('|')
        assert(len(code)==3)
        name = code[1]
        uniport_dict[name] = count
        count += 1
    
    with open(index_file, 'r') as f:
        data = f.read()
        data = data.split("\n")
        data = data[1:]
        all_data = []
        glycan_dict = {}
        
        glycan_count = 0
        glycan_list = []
        glycan_amount_dict = {}
        parser = PDBParser(PERMISSIVE = True, QUIET = True)
        error_point_list = []
        succed_point_list = cfg.succed_point_list
        for item in tqdm.tqdm(data):
            if item == "":
                break
            code = item.split(',')
            uniprot_id = code[0]
            # 没有在uniport文件中找到该蛋白
            if uniprot_id not in uniport_dict.keys():
                print('Warning: Unknown uniport:'+uniprot_id)
                continue
            site = int(code[1])
            glycan_type = code[2]
            file_info = code[3].split('_')
            if len(file_info)==1:
                file_info = code[8].split('-')
            name = file_info[0]
            chain = file_info[1]

            seq = uniport_list[uniport_dict[uniprot_id]].seq
            # uniport 序列长度不对
            if len(seq)<site:
                print('Warning: Wrong uniport sequence length:'+uniprot_id)
                continue
            marker_peptide = str(seq[site-1:site+4])

            # 已经确定该蛋白上找不到对应位点
            pdb_name = data_cfg.pdb_root + 'pdb' + name + '.ent'
            if not os.path.isfile(pdb_name):
                continue
            if (name+'_'+str(site) in error_point_list):
                continue
            if not (name+'_'+str(site) in succed_point_list):
                data = parser.get_structure(name, pdb_name)         
                aimchain = data[0][chain]         
                aim_pepti = []          
                for res in marker_peptide:             
                    aim_pepti.append(cfg.res_dict[res])          
                res_list =list(aimchain)         
                aim_res = None         
                for i in range(len(res_list)-5):             
                    if res_list[i].get_resname() == aim_pepti[0]:                 
                        if res_list[i+1].get_resname() == aim_pepti[1] and res_list[i+2].get_resname() == aim_pepti[2] and res_list[i+3].get_resname() == aim_pepti[3] and res_list[i+4].get_resname() == aim_pepti[4]:                     
                            aim_res = res_list[i]                     
                            break                  
                if not aim_res:
                    error_point_list.append(name+'_'+str(site))             
                    continue
                else:
                    succed_point_list.append(name+'_'+str(site))
            """
            if init:
                if glycan_type in glycan_dict.keys():
                    type_id = glycan_dict[glycan_type]
                    glycan_amount_dict[glycan_type] += 1
                else:
                    type_id = glycan_count
                    glycan_count += 1
                    glycan_dict[glycan_type] = type_id
                    glycan_amount_dict[glycan_type] = 1
                    glycan_list.append(glycan_type)
            
            else:
                if glycan_type in cfg.glycan_types.keys():
                    type_id = cfg.glycan_types[glycan_type]
                else:
                    type_id = cfg.glycan_types['UNKNOWN']
            """
            # 区分高甘露糖
            #
            is_MAN = False
            for i in range(len(glycan_type)):
                if glycan_type[i]=='N' and i<len(glycan_type)-1:
                    if(int(glycan_type[i+1]) == 2):
                        is_MAN = True
                    break

            find = False
            for item in all_data:
                if item.name == name and item.site == site and item.chain==chain:
                    item.label.append(is_MAN)
                    find = True
                    break
            
            if find:
                continue
            else:
                new_sample = SampleItem(name, site, marker_peptide, chain, [is_MAN])
                all_data.append(new_sample)
        """
        if init:
            rare_glycan = []
            for glycan,amount in glycan_amount_dict.items():
                if amount<data_cfg.amount_th:
                    rare_glycan.append(glycan)
            
            new_glycan_dict = {}
            count = 0
            for key in glycan_dict.keys():
                if key in rare_glycan:
                    continue
                new_glycan_dict[key] = count
                count += 1
            
            new_glycan_dict['UNKNOWN'] = count
            for item in all_data:
                for i in range(len(item.label)):
                    if glycan_list[item.label[i]] in rare_glycan:
                        item.label[i] = count
                    else:
                        item.label[i] = new_glycan_dict[glycan_list[item.label[i]]]
            print(succed_point_list)
            print(new_glycan_dict)
            print(rare_glycan)
        
        """
        type_0 = 0
        type_1 = 0
        type_2 = 0
        for item in all_data:
            if True in item.label and False in item.label:
                item.label = 0
                type_0+=1
            elif True in item.label:
                item.label = 1
                type_1+=1
            else:
                item.label = 2
                type_2+=1
        print(type_0)
        print(type_1)
        print(type_2)
        """
        if init:
            L = count + 1
        else:
            L = cfg.glycan_types_amount

        weights = torch.zeros((1, L))
        for item in all_data:
            label = [0.0]*L
            for glycan in item.label:
                label[glycan] = 1.0
            weights += torch.tensor(label)
        
        # 
        #weights = 1 / (weights / torch.min(weights))
        # 
        pos_weights = weights
        neg_weights = torch.tensor([len(all_data)]*L) - pos_weights
        
        weights = neg_weights/pos_weights
        """

        return all_data
    
def collatef(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def get_lr_step(lr, lr_decay, schedule, epochs):
    return (lr-lr*lr_decay)/(epochs*schedule[1]-epochs*schedule[0])

def accuracy_multilabel(outp, target):
    output = outp.clone()
    assert output.shape == target.shape, \
        "shapes output: {}, and target:{} are not aligned. ".\
            format(output.shape, target.shape)
    output.sigmoid_()
    return torch.round(output).eq(target).sum().cpu().numpy()/target.numel()

def accuracy(output, target):     
    acc = (output.argmax(1) == target).float().mean()     
    return acc.item()

def train_model(train_loader, model, use_cuda, lossf, opt, epoch, logger, logtrain):
    model.train()
    for g, target in train_loader:
        if use_cuda!='cpu':
            g = g.to(use_cuda)
            target = target.to(use_cuda)
        n = g.ndata['feature']
        e = g.edata['x']
        x = g.ndata['x']
        n, e, x, target = Variable(n), Variable(e), Variable(x),Variable(target)
        opt.zero_grad()
        output = model(g, n, e)
        #print(output)
        #print(target)
        loss = lossf(output, target)
        ACC = accuracy(output, target)
        #ACC = 0
        L = target.shape[0]
        logger.update_stat_value('train_loss', loss.item(),L)
        logger.update_stat_value('train_acc', ACC, L)
        loss.backward()
        opt.step()
    
    loss_value = logger.get_stat_value('train_loss')
    ACC_value = logger.get_stat_value('train_acc')
    logger.write_value(logtrain, [ACC_value, loss_value])

    print('Epoch: [{0}/{1}] Train Avg Loss {loss_value:.3f}; Train Avg ACC {ACC_value:.3f};'.format(epoch, train_cfg.epochs, loss_value=loss_value, ACC_value=ACC_value))
    logger.clear_stat_log()
    return ACC_value, loss_value

def valid_model(valid_loader, model, use_cuda, lossf, epoch, logger, logvalid):
    model.eval()
    for g, target in valid_loader:
        if use_cuda!='cpu':
            g = g.to(use_cuda)
            target = target.to(use_cuda)
        n = g.ndata['x']
        e = g.edata['x']
        #print(g)

        n, e, target = Variable(n), Variable(e), Variable(target)
        output = model(g, n, e)
        
        #print(output)
        #print(target)
        loss = lossf(output, target)
        ACC = accuracy(output, target)
        L = target.shape[0]

        logger.update_stat_value('test_loss', loss.item(), L)
        logger.update_stat_value('test_acc', ACC, L)
        
    
    loss_value = logger.get_stat_value('test_loss')
    acc_value = logger.get_stat_value('test_acc')
    logger.write_value(logvalid, [loss_value, acc_value])
    print('Epoch: [{0}/{1}] Valid Avg Loss {loss_value:.3f}; Valid Avg ACC {ACC_value:.3f};'.format(epoch, train_cfg.epochs, loss_value=loss_value, ACC_value=acc_value))
    logger.clear_stat_log()

    return acc_value, loss_value

def test_model(test_loader, model, use_cuda, lossf, logger):
    model.eval()
    for g, target in test_loader:
        if use_cuda!='cpu':
            g = g.to(use_cuda)
            target = target.to(use_cuda)
        n = g.ndata['x']
        e = g.edata['x']
        n, e, target = Variable(n), Variable(e), Variable(target)
        
        output = model(g, n, e)
        loss = lossf(output, target)
        #print(output)
        #print(target)
        ACC = accuracy(output, target)
        L = target.shape[0]
        
        logger.update_stat_value('test_loss', loss.item(), L)
        logger.update_stat_value('test_acc', ACC, L)
        
    
    loss_value = logger.get_stat_value('test_loss')
    acc_value = logger.get_stat_value('test_acc')
    logger.clear_stat_log()

    return acc_value, loss_value

def save_model(model, ACC, path, filename):
    if not os.path.isdir(path):
        os.mkdir(path)
    
    file = path + filename
    torch.save({'state_dict':model.state_dict(), 'ACC':ACC}, file)
