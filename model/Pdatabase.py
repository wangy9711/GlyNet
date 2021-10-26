import os
from dgl.utils.internal import zero_index
import numpy as np
from numpy import sqrt
import torch.utils.data as data

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import is_aa
import dgl
from dgl import DGLGraph
from dgl.data.utils import save_graphs, load_graphs
import torch
from Bio.PDB.DSSP import DSSP


from .Config import DataConfig, ModelConfig
from .Putils import get_topK


cfg = ModelConfig()
data_cfg = DataConfig()

class Atom:
    def __init__(self, name=None, x=None, y=None, z=None, chain=None, ACC=None):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.chain = chain
        self.ACC = ACC

def Euc_dis_atom(atom1, atom2):
    return sqrt((atom1.x-atom2.x)*(atom1.x-atom2.x)+(atom1.y-atom2.y)*(atom1.y-atom2.y)+(atom1.z-atom2.z)*(atom1.z-atom2.z))

def Euc_dis(x1,x2,y1,y2,z1,z2):
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))


class ProteinBase(data.Dataset):
    def __init__(self, raw_datas, init=False):
        super(ProteinBase, self).__init__()
        self.raw_datas = raw_datas
        self.r = data_cfg.r
        self.top_k = data_cfg.top_k
        self.pdb_root = data_cfg.pdb_root
        self.graph_root = data_cfg.graph_root
        self.count = 0
        if not os.path.isdir(self.graph_root):
            os.mkdir(self.graph_root)

        self.parser = PDBParser(PERMISSIVE = True, QUIET = True)
        if init:
            print("Init Database ... ")
            self.init_database()
            print("Init finished!")
    
    def __getitem__(self, index):
        item = self.raw_datas[index]
        graph_name = self.graph_root+item.name+'_'+item.chain+'_'+str(item.site)+'.graph'
        graph = load_graphs(graph_name)
        label = [0.0]*3
        #for glycan in item.label:
        #    label[glycan] = 1.0
        label[item.label] = 1.0
        return graph[0][0], item.label
    
    def init_database(self):
        for item in self.raw_datas:
            self.pdb2graph(item)
    
    def pdb2graph(self, sample):
        protein_name = sample.name
        pdb_name = self.pdb_root+'pdb'+protein_name+'.ent'
        graph_name = self.graph_root+protein_name+'_'+sample.chain+'_'+str(sample.site)+'.graph'
        if os.path.isfile(graph_name):
            return

        data = self.parser.get_structure(protein_name,pdb_name)
        #assert(len(data)==1)
        chain = data[0][sample.chain]
        aim_pepti = []

        for res in sample.marker_peptide:
            aim_pepti.append(cfg.res_dict[res])

        res_list =list(chain)
        aim_res = None
        for i in range(len(res_list)-5):
            if res_list[i].get_resname() == aim_pepti[0]:
                if res_list[i+1].get_resname() == aim_pepti[1] and res_list[i+2].get_resname() == aim_pepti[2] and res_list[i+3].get_resname() == aim_pepti[3] and res_list[i+4].get_resname() == aim_pepti[4]:
                    aim_res = res_list[i]
                    break
        
        if not aim_res:
            self.raw_datas.remove(sample)
            return 
        
        tags = aim_res.get_id()
        aim_atom = aim_res['CA']
        aim_x = aim_atom.get_vector()[0]
        aim_y = aim_atom.get_vector()[1]
        aim_z = aim_atom.get_vector()[2]
        center_atom = Atom(aim_res.get_resname(), aim_x, aim_y, aim_z, sample.chain, tags)
        #from 'CA' to aim_res.get_resname()
        all_atom = []
        all_atom.append(center_atom)
       # for chain in data[0].get_list():
       #     chain_name = chain.get_id()
       #     for residue in chain.get_list():
       #         if not is_aa(residue):
       #             continue
       #         tags = residue.get_full_id()
       #         if tags[3][0] != " ":
       #             continue
       #         for atom_item in residue.get_list():
       #             x = atom_item.get_vector()[0]
       #             y = atom_item.get_vector()[1]
       #             z = atom_item.get_vector()[2]
       #             if Euc_dis(x,aim_x,y,aim_y,z,aim_z)>data_cfg.r:
       #                 continue
       #             tatom = Atom(atom_item.get_name(), x,y,z,chain_name)
       #             all_atom.append(tatom)
        
        # find all atom where dis(atom, center)<30
        for chain in data[0].get_list():
            chain_name = chain.get_id()
            for residue in chain.get_list():
                if not is_aa(residue):
                    continue
                tags = residue.get_full_id()
                if tags[3][0] != " ":
                    continue
                res_name = residue.get_resname()
                try:
                    x = residue['CA'].get_vector()[0]
                    y = residue['CA'].get_vector()[1]
                    z = residue['CA'].get_vector()[2]
                except:
                    continue
                if Euc_dis(x,aim_x,y,aim_y,z,aim_z)>data_cfg.r:
                    continue
                tags = residue.get_id()
                tatom = Atom(res_name, x,y,z,chain_name, tags)
                all_atom.append(tatom)
        L = len(all_atom)
        dis_map = np.zeros((L,L))
        for i in range(L-1):
            dis_map[i][i] = 0
            j =i+1
            while(j<L):
                dis = Euc_dis_atom(all_atom[i], all_atom[j])
                dis_map[i][j] = dis
                dis_map[j][i] = dis
                j +=1
            
        # get distance map
        top = min(data_cfg.top_k+1, L-1)
        top_k_index = get_topK(dis_map,top, 0)
        # shape (topk+1)*L
        # get top k

        # create graph
        g = DGLGraph()
        g_node_count = 0
        g_edge_count = 0
        dssp = DSSP(data[0], pdb_name)
        for i in range(L):
            g.add_nodes(1)
            feature = torch.zeros((1, cfg.res_amount+data_cfg.acc_bins))
            feature[0][cfg.res_types[all_atom[i].name]] = 1
            x = torch.zeros((1,3))
            x[0][0] = all_atom[i].x
            x[0][1] = all_atom[i].y
            x[0][2] = all_atom[i].z
            
            try:
                this_acc = dssp[(all_atom[i].chain, all_atom[i].ACC)][3]
            except:
                this_acc = 0

            this_bin = int(this_acc*data_cfg.acc_bins)
            if(this_bin>=data_cfg.acc_bins):
                this_bin = data_cfg.acc_bins-1
            feature[0][cfg.res_amount+this_bin] = 1

            #feature = torch.zeros((1, cfg.atom_types_amount))
            # feature size 1*1 原子索引
            #feature[0][cfg.atom_types[all_atom[i].name]] = 1
            g.nodes[[g_node_count]].data['feature'] = feature
            g.nodes[[g_node_count]].data['x'] = x
            g_node_count += 1   
        
        edge_list = []
        for i in range(L):
            for j in range(top):
                if top_k_index[j][i] == i:
                    continue
                if dis_map[i][top_k_index[j][i]]>data_cfg.r:
                    continue
                
                if (i, top_k_index[j][i]) not in edge_list:
                    g.add_edge(i, top_k_index[j][i])
                    edge_list.append((i, top_k_index[j][i]))
                    #feature size 1*N, onehot
                    feature = torch.zeros((1, data_cfg.bins))
                    gap = data_cfg.r/data_cfg.bins
                    feature[0][int(dis_map[i][top_k_index[j][i]]/gap)] = 1
                    g.edges[[g_edge_count]].data['x'] = feature
                    g_edge_count += 1
                if (top_k_index[j][i],i) not in edge_list:
                    g.add_edge(top_k_index[j][i],i)
                    edge_list.append((top_k_index[j][i],i))
                    #feature size 1*N, onehot
                    feature = torch.zeros((1, data_cfg.bins))
                    gap = data_cfg.r/data_cfg.bins
                    feature[0][int(dis_map[i][top_k_index[j][i]]/gap)] = 1
                    g.edges[[g_edge_count]].data['x'] = feature
                    g_edge_count += 1


        
        save_graphs(graph_name, g)
    
    def __len__(self):
        return len(self.raw_datas)
        







