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
from utils import *
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
            # pdb.set_trace()
            for index in range(3):
                print("process tCNNs ",index)
                data_list_tCNN_split = data_list_tCNN[index*int((len(data_list_tCNN)/3)):(index+1)*int((len(data_list_tCNN)/3))]
                data, slices = self.collate(data_list_tCNN_split)
                torch.save((data, slices), save_root_dir.split(".pt")[0]+"_tCNN_{}.pt".format(index))
        else:
            torch.save((data, slices), save_root_dir.split(".pt")[0]+"_tCNN.pt")

                    
        data, slices = self.collate(data_list_DeepTTC)
        torch.save((data, slices), save_root_dir.split(".pt")[0]+"_DeepTTC.pt")
    
    
def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "data/"
#folder = ""

def load_drug_list():
    filename = folder + "Druglist.csv"
    csvfile = open(filename, "rb")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs

def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(folder + 'pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)

def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def load_cid_dict():
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k,v in cids_dict.iteritems()]
    inv_cids_dict = {v:k for k,v in cids_dict.iteritems()}
    download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES', overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()

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
                # print(smile, allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))
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
    reader = csv.reader(open("/xxx/GDSCv2/drug_smiles.csv"))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
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

def save_cell_mut_matrix():
    f = open("/xxx/PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
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
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)
    
    return cell_dict, cell_feature


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""
def save_mix_drug_cell_matrix():
    f = open("/xxx/GDSCv2/PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []
    random.shuffle(temp_data)
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            xd.append(drug_smile[drug_dict[drug]])
            xc.append(cell_feature[cell_dict[cell]])
            y.append(ic50)
            bExist[drug_dict[drug], cell_dict[cell]] = 1
            lst_drug.append(drug)
            lst_cell.append(cell)
        
    with open('drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)

    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)

    with open('list_drug_mix_test', 'wb') as fp:
        pickle.dump(lst_drug[size1:], fp)
        
    with open('list_cell_mix_test', 'wb') as fp:
        pickle.dump(lst_cell[size1:], fp)

    xd_train = xd[:size]
    xd_val = xd[size:size1]
    xd_test = xd[size1:]

    xc_train = xc[:size]
    xc_val = xc[size:size1]
    xc_test = xc[size1:]

    y_train = y[:size]
    y_val = y[size:size1]
    y_test = y[size1:]

    dataset = 'GDSCv2'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    train_data = Dataset_pan_mut(root='/xxx/data', dataset=dataset+'_train', xd=xd_train, xt=xc_train, y=y_train, smile_graph=smile_graph)
    val_data = Dataset_pan_mut(root='/xxx/data', dataset=dataset+'_val', xd=xd_val, xt=xc_val, y=y_val, smile_graph=smile_graph)
    test_data = Dataset_pan_mut(root='/xxx/data', dataset=dataset+'_test', xd=xd_test, xt=xc_test, y=y_test, smile_graph=smile_graph)


def seed_torch(seed=532):
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
    
if __name__ == "__main__":
    seed_torch()
    save_mix_drug_cell_matrix()
