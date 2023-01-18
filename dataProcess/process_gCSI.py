

import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx

import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
import torch,pdb
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from torch_geometric.data import Data
from sklearn.decomposition import PCA, KernelPCA
from tqdm import tqdm

from scipy import optimize
import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch,pdb
import matplotlib.pyplot as plt
from tqdm import tqdm


class Dataset_pan_mut(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None,saliency_map=False):

        #root is required for save preprocessed data, default is '/tmp'
        super(Dataset_pan_mut, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        
        data_list_normal = []
        data_list_tCNN = []
        data_list_DeepTTC = []
        
        data_len = len(xd)
        
        smiles_list = []
        for i in range(data_len):
            smiles_list.append(xd[i])
        
        smiles_list_set = set(smiles_list)
        smiles_dict = {k:i for i,k in enumerate(smiles_list_set)}

        canonical = getTCNNsMatrix(list(smiles_list_set))

        print('Converting data to DATA class.')
        for i in tqdm(range(data_len)):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            tCNNs_drug_matrix = canonical[smiles_dict[smiles]]
            # convert SMILES to molecular representation using rdkit
            # pdb.set_trace()
            c_size, features, edge_index, edge_attr, encode_TTC  = smile_graph[smiles]
            
            
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(np.array(features)),            
                                edge_index=torch.LongTensor(edge_index),
                                edge_attr = torch.LongTensor(edge_attr),
                                smiles = smiles,
                                y=torch.FloatTensor([labels]))
 
            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target = torch.tensor([target], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            
            # append graph, label and target sequence to data list
            data_list_normal.append(GCNData)
          
            GCNData = DATA.Data(smiles = smiles,
                                tCNNs_drug_matrix = tCNNs_drug_matrix,
                                y=torch.FloatTensor([labels]))
            if self.saliency_map == True:
                GCNData.target = torch.tensor([target], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list_tCNN.append(GCNData)
            
            GCNData = DATA.Data(DeepTTC_drug_encode=torch.Tensor(encode_TTC[0]).reshape(1,-1),
                                DeepTTC_drug_encode_mask=torch.Tensor(encode_TTC[1]).reshape(1,-1),              
                                smiles = smiles,
                                y=torch.FloatTensor([labels]))
            if self.saliency_map == True:
                GCNData.target = torch.tensor([target], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
                        
            data_list_DeepTTC.append(GCNData)

        if self.pre_filter is not None:
            data_list_normal = [data for data in data_list_normal if self.pre_filter(data)]
            data_list_tCNN = [data for data in data_list_tCNN if self.pre_filter(data)]
            data_list_DeepTTC = [data for data in data_list_DeepTTC if self.pre_filter(data)]

        

        if self.pre_transform is not None:
            data_list_normal = [self.pre_transform(data) for data in data_list_normal]
            data_list_tCNN = [self.pre_transform(data) for data in data_list_tCNN]
            data_list_DeepTTC = [self.pre_transform(data) for data in data_list_DeepTTC]

        print('Graph construction done. Saving to file.')

        save_root_dir = self.processed_paths[0]
        
        
        
        data, slices = self.collate(data_list_normal)
        torch.save((data, slices), save_root_dir.split(".pt")[0]+"_normal.pt")

        if len(data_list_tCNN)>50000:
            print("len(data_list_tCNN)={}".format(len(data_list_tCNN)))
            # pdb.set_trace()
            for index in range(5):
                print("process tCNNs ",index)
                if index<4:
                    data_list_tCNN_split = data_list_tCNN[index*int((len(data_list_tCNN)/5)):(index+1)*int((len(data_list_tCNN)/5))]
                elif index ==4:
                    data_list_tCNN_split = data_list_tCNN[index*int((len(data_list_tCNN)/5)):]
                data, slices = self.collate(data_list_tCNN_split)
                torch.save((data, slices), save_root_dir.split(".pt")[0]+"_tCNN_{}.pt".format(index))
        else:
            torch.save((data, slices), save_root_dir.split(".pt")[0]+"_tCNN.pt")

                    
        data, slices = self.collate(data_list_DeepTTC)
        torch.save((data, slices), save_root_dir.split(".pt")[0]+"_DeepTTC.pt")

'''tCNNs drug smile matrix'''
import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math

def getTCNNsMatrix(smiles):
    c_chars, _ = charsets(smiles)
    c_length = max(map(len, map(string2smiles_list, smiles)))
    
    smiles = np.array(smiles)
    count = smiles.shape[0]

    smiles = [string2smiles_list(smiles[i]) for i in range(count)]
    
    canonical = smiles_to_onehot(smiles, c_chars, c_length)
    
    save_dict = {}
    save_dict["canonical"] = canonical
    save_dict["c_chars"] = c_chars
    
    return  canonical

def charsets(smiles):
    from functools import reduce
    union = lambda x, y: set(x) | set(y)
    c_chars = list(reduce(union, map(string2smiles_list, smiles)))
    i_chars = list(reduce(union, map(string2smiles_list, smiles)))
    return c_chars, i_chars

def string2smiles_list(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i-1:i+1])
            i += 1
        else:
            char_list.append(string[i-1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list

def smiles_to_onehot(smiles, c_chars, c_length):
    print("Onehot encode...")
    c_ndarray = np.ndarray(shape=(len(smiles), len(c_chars), c_length), dtype=np.float32)
    for i in tqdm(range(len(smiles))):
        c_ndarray[i, ...] = onehot_encode(c_chars, smiles[i], c_length)
    return c_ndarray

def onehot_encode(char_list, smiles_string, length):
    
    encode_row = lambda char: map(int, [c == char for c in smiles_string])
    
    ans = [ list(x) for x in map(encode_row, char_list)]
    ans = np.array(ans)
    if ans.shape[1] < length:
        
        residual = np.zeros((len(char_list), length - ans.shape[1]), dtype=np.int8)
        ans = np.concatenate((ans, residual), axis=1)
    return ans

def cell_line_PanMut():
    f = open("/xxx/PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    cell_name_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col
            
        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
            
        if is_mutated == 1:
            matrix_list.append((row, col))


        cell_name_dict[str(item[1])] = item[0]
        
        
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1
   
    return cell_dict, cell_feature, cell_name_dict


def f_1(x, A, B):
    return A * x + B


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

"""
The following code will convert the SMILES format into onehot format
"""

def atom_features(atom):
    # print(atom.GetChiralTag())
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding(atom.GetChiralTag(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()]+[atom.GetAtomicNum()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


import codecs
from subword_nmt.apply_bpe import BPE

def drug2emb_encoder(smile):
    vocab_path = "/xxx/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("/xxx/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)
    
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    # Chem.MolFromSmiles('CC[C@H](F)Cl', useChirality=True)
    # mol = Chem.MolFromSmiles(smile, useChirality=True, isomericSmiles= True, kekuleSmiles = True)
    # pdb.set_trace()
    c_size = mol.GetNumAtoms()
    
    features = []
    node_dict = {}
    
    for atom in mol.GetAtoms():
        node_dict[str(atom.GetIdx())] = atom
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    
    # bonds
    num_bond_features = 5
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # if allowable_features['possible_bond_dirs'].index(bond.GetBondDir()) !=0:
            # pdb.set_trace()
                # print(allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))
            edge_feature = [
                bond.GetBondTypeAsDouble(), 
                bond.GetIsAromatic(),
                # 芳香键
                bond.GetIsConjugated(),
                # 是否为共轭键
                bond.IsInRing(),             
                allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    
        
    '''DeepTTC'''
    encode_TTC = drug2emb_encoder(smile)
    
    return c_size, features, edge_index, edge_attr, encode_TTC

def load_drug_smile():
    

    reader = csv.reader(open("/xxx/gCSI/drug_smiles.csv"))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        # pdb.set_trace()
        # print(item)
        name = item[0]
        smile = item[3]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    
    return drug_dict, drug_smile, smile_graph

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    
def CTRP_IC50():
    seed_torch(42)
    
    IC50_dir = '/xxx/gCSI/output/gCSI_GRmetrics_v1.3.tsv'

    IC50_data = pd.read_csv(
        IC50_dir,
        sep='\t',
        header=0,
    )

    cell_line_drug_mp = {}
    for index in tqdm(range(len(IC50_data))):
        if str(IC50_data['GR50'].loc[index]) == 'nan' or  str(IC50_data['GR50'].loc[index]) == 'inf':
            continue
        key = IC50_data['DrugName'].loc[index] +'*'+ IC50_data['CellLineName'].loc[index] 
        cell_line_drug_mp[key] = {}
        cell_line_drug_mp[key]['ic50']= IC50_data['GR50'].loc[index]
        

    # 从GDSC Cell_line 中查找CTRP 出现过的Cell line Name
    cell_dict, cell_feature, cell_name_dict = cell_line_PanMut()
    drug_dict, drug_smile, smile_graph = load_drug_smile()
    
    drug_dict_lower = {str.lower(k):v for k,v in drug_dict.items()}
    
    cell_name_dict_re = {str.lower(value):key for key,value in cell_name_dict.items()}

    # 回到正轨，构建dataset
    xd = []
    xc = []
    y = []
    temp_data = []
    print("Total IC50 info is {}...".format(len(cell_line_drug_mp.items())))
    
    
    for key, value in tqdm(cell_line_drug_mp.items()):
        
        drug_name, cell_line_name = key.split("*")
        
        drug_name = str.lower(drug_name)
        cell_line_name = str.lower(cell_line_name)
        
        
        if cell_name_dict_re.get(cell_line_name) is None:
            continue

        if drug_dict_lower.get(drug_name) is None:
            continue
        
        ic50 = float(value['ic50']) 
        
        if ic50 <0:
            continue
        else:
            ic50 = math.log(ic50*1000000)
            ic50 = 1 / (1 + math.exp( -0.1 * float(ic50)))
            temp_data.append([
                cell_feature[cell_dict[cell_name_dict_re.get(cell_line_name)]],
                drug_smile[drug_dict_lower[drug_name]],
                ic50
            ])

    random.shuffle(temp_data)
    
    
    for item in tqdm(temp_data):
        xc.append(item[0])
        xd.append(item[1])
        y.append(item[2])

    
    print("After process, there is {}".format(len(y)))
    
    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)
    
    xd_train = xd
    xc_train = xc
    y_train = y


    
    # pdb.set_trace()
    dataset = 'gCSI'
    print('preparing ', dataset + ' !')

    train_data = Dataset_pan_mut(root='/xxx/gCSI', dataset=dataset+'_All', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)

        
CTRP_IC50() 