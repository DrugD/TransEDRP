import pdb, torch
import numpy as np
from tqdm import tqdm
from models.gat_gcn_transformer import GAT_GCN_Transformer
from rdkit import Chem
import networkx as nx
import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
from torch_geometric import data as DATA
import random

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

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    # Chem.MolFromSmiles('CC[C@H](F)Cl', useChirality=True)
    # mol = Chem.MolFromSmiles(smile, isomericSmiles= True), kekuleSmiles = True)
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
            # print(smile,allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))
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
    

    return c_size, features, edge_index, edge_attr

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
    
    c_ndarray = np.ndarray(shape=(len(smiles), len(c_chars), c_length), dtype=np.float32)
    for i in range(len(smiles)):
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

def load_cell_mut_matrix(cell_lines_path):
    f = open(cell_lines_path)
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
    
    return cell_dict, cell_feature
            
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
    
def save2Txt(data,save_txt_path):
    data=np.array(data)
    # np.save(save_txt_path,data)   # 保存为.npy格式
    np.savetxt(save_txt_path, data, delimiter=',', fmt='%.2f')
        
def main(cell_lines_path, drug_smils, model_pth, record_txt_save_path):
    cuda_name = 'cuda'

    cell_dict, cell_features = load_cell_mut_matrix(cell_lines_path)

    cell_dict = {value:key for key,value in cell_dict.items()}


    f=open(drug_smils, encoding='gbk')
    drug_list=[]
    for line in f:
        drug_list.append(line.strip())

    canonical = getTCNNsMatrix(drug_list)
    
    max_record = []
    
    for j in tqdm(range(len(cell_features))):
        record = []
        for i in range(len(drug_list)):
           
            smiles = drug_list[i]
            tCNNs_drug_matrix = canonical[i]
            cell_feature = cell_features[j]
            
            c_size, features, edge_index, edge_attr = smile_to_graph(smiles)

            # pdb.set_trace()

            GCNData = DATA.Data(x=torch.Tensor(np.array(features)),
                                    edge_index=torch.LongTensor(edge_index),
                                    edge_attr = torch.LongTensor(edge_attr),
                                    smiles = smiles,
                                    tCNNs_drug_matrix = tCNNs_drug_matrix)
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData.target = torch.FloatTensor(np.array([cell_feature]))

            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

            model = GAT_GCN_Transformer()
            model.load_state_dict(torch.load(model_pth), strict=True)
            model = model.to(device)

            GCNData = GCNData.to(device)
            IC50_pred, _ = model.infer(GCNData)
            # print(smiles,cell_dict[j],IC50_pred.item())
            IC50_pred = math.log(1/pow((1/IC50_pred.item()-1),10))
            record.append(IC50_pred)
        
        max_record.append([j,record[0],record[1],abs(record[0]-record[1]),record[2],record[3],abs(record[2]-record[3])])
    save2Txt(max_record, record_txt_save_path)
    
if __name__ == "__main__":
    cell_lines_path = "data/PANCANCER_Genetic_feature.csv"
    drug_smils = "data/infer_drug.txt"
    model_pth = "exp/tsf_edge_type_4_15_15_15_sxDate_with_transformer_concat_h_1/model_GAT_GCN_Transformer_GDSC.model"
    record_txt_save_path = "infer/IsomericRecord/test3.txt"
    
    seed_torch(171)
    main(cell_lines_path,drug_smils,model_pth,record_txt_save_path)