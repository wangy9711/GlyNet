import sys
import os
import torch

from torch.autograd import Variable

from model.GlyNet import GlyNet
from model.IUPAC2graph import get_data_from_code
from model.Config import ModelConfig, TestConfig, DataConfig


data_cfg = DataConfig()
model_cfg = ModelConfig()
test_cfg = TestConfig()


if __name__ == '__main__':
    if len(sys.argv)!=3 :
        raise ValueError('Please enter valid parameters!')
    
    input_type = sys.argv[1]
    input_seq = ''
    if input_type == 'file':
        input_file = sys.argv[2]
        if os.path.isfile(input_file):
            input_seq = open(input_file, 'r').readline()
        else:
            raise ValueError('Input file does not exist!')
    
    elif input_type == 'sequence':
        input_seq=sys.argv[2]
    
    else:
        raise ValueError('Please select a correct input format! (\'file\' or \'sequence\')')
    


    input_graph = get_data_from_code(input_seq, data_cfg.glycan, data_cfg.link, data_cfg.monose)

    edge_size = len(data_cfg.link)
    node_type = len(data_cfg.glycan)
    target_size = 1
    message_embedding_size = model_cfg.message_embedding_size
    graph_embedding_size = model_cfg.graph_embedding_size
    mid_embedding_size = model_cfg.mid_embedding_size
    layers = model_cfg.layers
    dropouts = model_cfg.dropout
    residual = model_cfg.residual

    model = GlyNet(
            edge_feature_size = edge_size, 
            node_type=node_type, 
            target_size=target_size, 
            message_size=message_embedding_size, 
            graph_size = graph_embedding_size,
            mid_size = mid_embedding_size,
            layers=layers,
            dropout = dropouts,
            residual=residual)
    
    
    checkpoint = torch.load(test_cfg.model_file)
    model.load_state_dict(checkpoint['state_dict'])

    if test_cfg.use_cuda and torch.cuda.is_available():
        use_cuda = 'cuda:' + test_cfg.use_cuda
    else:
        use_cuda = 'cpu'
    if use_cuda != 'cpu':
        model = model.to(use_cuda)
    

    if use_cuda!='cpu':
        input_graph = input_graph.to(use_cuda)
    n = input_graph.ndata['x'].long().squeeze(dim=1)
    e = input_graph.edata['x']
    n, e, = Variable(n), Variable(e)

    output = model(input_graph, n, e)

    print(f"The probability that this glycan is immunogenic: {output}")

    





