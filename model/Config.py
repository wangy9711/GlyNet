class ModelConfig:
    def __init__(self):
        self.message_embedding_size = 256
        self.graph_embedding_size = 128
        self.mid_embedding_size = 64
        self.layers = 5
        self.dropout = 0.4
        self.residual = True


class TrainConfig:
    def __init__(self):
        self.train_times = 100
        self.datasetPath = './data/data_label.csv'
        self.logPath = './log/'
        self.load_checkpoint = False
        self.checkpoint_path = './checkpoint/'
        self.checkpoint_file = 'checkpoint.pth'
        self.batch_size = 32
        self.use_cuda = '0'
        self.epochs = 50
        self.lr = 2.5e-4
        self.lr_decay = 0.8 
        self.lr_schedule = [0.1, 0.9]
        self.log_interval = 1
        self.train_log_file = 'trainlog.csv'
        self.valid_log_file = 'validlog.csv'
        self.test_log_file = 'testlog.csv'

        self.valid_num = 100
        self.test_num = 100
        

class TestConfig:
    def __init__(self):
        self.model_file = './checkpoint/cpt.pth'
        self.use_cuda = False


class DataConfig:
    def __init__(self):
        self.use_monose_info = False
        self.use_link_inof = True
        self.use_reduce_info = False
        #label <5
        self.glycan = ['NeuNAc', 'Gal', 'GlcNAc', 'Man', 'Fuc', 'GalNAc', 'GlcNAcOS', 'Xyl', 'GalOS', 'Glc', 'ManOP', 'GalNAcOS', 'NeuNGc', 'Rha', 'FucNAc', 'QuiNAcNBut', 'Col', 'GalA', 'QuiNAc', 'ManNAcA', 'QuiNAlaAc', 'GalNAcA', 'GalOAc', 'Ribf', 'Sug', '6dTalOAc', 'ManNAc', 'Kdo', 'GalAGroN', 'ManOAc', 'RhaOAc', 'Galf', 'NeuNAcOGc', 'GlcNAcA', 'GlcN', 'RhaOMe', 'DDManHep', 'LDManHep', 'LDManHepOP', 'Abe', 'GlcNAcOAc', '6dTal', 'RhaNAc', 'FucNAm', 'GlcA', 'KdoOP', 'AraN', 'GalAAlaLys', 'Ara', 'GulNAcA', 'QuiNFo', 'QuiNBut', 'Tyv', 'GlcOAc', 'ManNAcNAmA', '6dAltf', 'Unknown']
        # lable <10
        self.link = ['a2-3', 'b1-4', 'b1-2', 'a1-6', 'a1-3', 'a1-2', 'b1-3', 'a2-6', 'b1-6', 'a1-4', 'a2-8', 'a1-5', 'b1-7', 'a1-7', 'a2-4', 'Unknown']

        self.monose = {
            'Unknown':[0, 0, 0, 0, 0, 0],
            'Hexose':[180.16, 6, 12, 0, 6, 0],
            'HexNAc':[221.21, 8, 15, 1, 6, 0],
            'Hexuronate':[194.14, 6, 10, 0, 7, 0],
            'DeoxyhexNAc':[205.21, 8, 15, 1, 5, 0],
            'Di-deoxyhexose':[148.16, 6, 12, 0, 4, 0],
            'Pentose':[150.13, 5, 10, 0, 5, 0],
            'Deoxynonulosonate':[268.22, 9, 16, 0, 9, 0],
            'Di-deoxynonulosonate':[250.25, 9, 18, 2, 6, 0],
            'Glc':[180.16, 6, 12, 0, 6, 0],
            'Man':[180.16, 6, 12, 0, 6, 0],
            'Gal':[180.16, 6, 12, 0, 6, 0],
            'Gul':[180.16, 6, 12, 0, 6, 0],
            'Alt':[180.16, 6, 12, 0, 6, 0],
            'Tal':[180.16, 6, 12, 0, 6, 0],
            'Ido':[180.16, 6, 12, 0, 6, 0],
            'GlcNAc':[221.21, 8, 15, 1, 6, 0],
            'ManNAc':[221.21, 8, 15, 1, 6, 0],
            'GalNAc':[221.21, 8, 15, 1, 6, 0],
            'GulNAc':[221.21, 8, 15, 1, 6, 0],
            'AltNAc':[221.21, 8, 15, 1, 6, 0],
            'AllNAc':[221.21, 8, 15, 1, 6, 0],
            'TalNAc':[221.21, 8, 15, 1, 6, 0],
            'IdoNAc':[221.21, 8, 15, 1, 6, 0],
            'GlcN':[179.17, 6, 13, 1, 5, 0],
            'ManN':[179.17, 6, 13, 1, 5, 0],
            'GalN':[179.17, 6, 13, 1, 5, 0],
            'GulN':[179.17, 6, 13, 1, 5, 0],
            'AltN':[179.17, 6, 13, 1, 5, 0],
            'AllN':[179.17, 6, 13, 1, 5, 0],
            'TalN':[179.17, 6, 13, 1, 5, 0],
            'IdoN':[179.17, 6, 13, 1, 5, 0],
            'GlcA':[194.14, 6, 10, 0, 7, 0],
            'ManA':[194.14, 6, 10, 0, 7, 0],
            'GalA':[194.14, 6, 10, 0, 7, 0],
            'GulA':[194.14, 6, 10, 0, 7, 0],
            'AltA':[194.14, 6, 10, 0, 7, 0],
            'AllA':[194.14, 6, 10, 0, 7, 0],
            'TalA':[194.14, 6, 10, 0, 7, 0],
            'IdoA':[194.14, 6, 10, 0, 7, 0],
            'Qui':[164.16, 6, 12, 0, 5, 0],
            'Rha':[164.16, 6, 12, 0, 5, 0],
            '6dGul':[164.16, 6, 12, 0, 5, 0],
            '6dAlt':[164.16, 6, 12, 0, 5, 0],
            '6dTal':[164.16, 6, 12, 0, 5, 0],
            'Fuc':[164.16, 6, 12, 0, 5, 0],
            'QuiNAc':[205.21, 8, 15, 1, 5, 0],
            'RhaNAc':[205.21, 8, 15, 1, 5, 0],
            '6dAltNAc':[205.21, 8, 15, 1, 5, 0],
            '6dTalNAc':[205.21, 8, 15, 1, 5, 0],
            'FucNAc':[205.21, 8, 15, 1, 5, 0],
            'Oli':[148.16, 6, 12, 0, 4, 0],
            'Tyv':[148.16, 6, 12, 0, 4, 0],
            'Abe':[148.16, 6, 12, 0, 4, 0],
            'Par':[148.16, 6, 12, 0, 4, 0],
            'Dig':[148.16, 6, 12, 0, 4, 0],
            'Col':[148.16, 6, 12, 0, 4, 0],
            'Ara':[150.13, 5, 10, 0, 5, 0],
            'Lyx':[150.13, 5, 10, 0, 5, 0],
            'Xyl':[150.13, 5, 10, 0, 5, 0],
            'Rib':[150.13, 5, 10, 0, 5, 0],
            'Kdn':[268.22, 9, 16, 0, 9, 0],
            'Neu5Ac':[309.27, 11, 19, 1, 9, 0],
            'Neu5Gc':[325.27, 11, 19, 1, 10, 0],
            'Neu':[267.23, 9, 17, 1, 8, 0],
            'Pse':[250.25, 9, 18, 2, 6, 0],
            'Leg':[250.25, 9, 18, 2, 6, 0],
            'Aci':[250.25, 9, 18, 2, 6, 0],
            '4eLeg':[250.25, 9, 18, 2, 6, 0],
            'Bac':[162.19, 6, 14, 2, 3, 0],
            'LDmanHep':[210.18, 7, 14, 0, 7, 0],
            'Kdo':[238.19, 8, 14, 0, 8, 0],
            'Dha':[222.15, 7, 10, 0, 8, 0],
            'DDmanHep':[210.18, 7, 14, 0, 7, 0],
            'MurNAc':[293.27, 11, 19, 1, 8, 0],
            'MurNGc':[309.27, 11, 19, 1, 9, 0],
            'Mur':[251.23, 9, 17, 1, 7, 0],
            'Api':[150.13, 5, 10, 0, 5, 0],
            'Fru':[180.16, 6, 12, 0, 6, 0],
            'Tag':[180.16, 6, 12, 0, 6, 0],
            'Sor':[180.16, 6, 12, 0, 6, 0],
            'Psi':[180.16, 6, 12, 0, 6, 0]
        }
