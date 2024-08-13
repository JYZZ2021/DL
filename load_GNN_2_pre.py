# 代码源自：华理小钊
# 时   间：2023/3/25 20:58
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from rdkit import Chem
from GNN3 import MolecularGraphNeuralNetwork

def extract_atoms_nodes(mol, atom_dict_path):
    """根据原子字典提取原子id"""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atom_dict = pd.read_csv(atom_dict_path)
    columns = atom_dict.columns
    atom_nodes = []
    for i in range(len(atoms)):
        atom_str = str(atoms[i])
        if atom_str in columns:
            atom_node = atom_dict[atom_str][0]
        else:
            atom_node = 999999
        atom_nodes.append(atom_node)
    return np.array(atom_nodes)

def extract_ijbonddict(mol, bond_dict_path):
    """根据键的字典提取键的id"""
    bond_dict = pd.read_csv(bond_dict_path)
    columns_bond_dict = bond_dict.columns
    i_jbond_dict = defaultdict(lambda: [])
    for bond in mol.GetBonds():
        bond_type_str = str(bond.GetBondType())
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bond_type_str in columns_bond_dict:
            bond_number = bond_dict[bond_type_str][0]
        else:
            bond_number = 1111111
        i_jbond_dict[i].append((j, bond_number))
        i_jbond_dict[j].append((i, bond_number))
    return i_jbond_dict

def extract_fingerprint(radius, atoms, i_jbond_dict, fingerprint_dict_path):
    """根据指纹字典提取指纹id"""
    fingerprint_dict = pd.read_csv(fingerprint_dict_path)
    columns_fingerprint_dict = fingerprint_dict.columns

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprint_str = str(fingerprint)
                if fingerprint_str in columns_fingerprint_dict:
                    nodes_.append(fingerprint_dict[fingerprint_str][0])
                else:
                    nodes_.append(22222222)

            nodes = nodes_
    return np.array(nodes)

if __name__ == '__main__':
    """参数"""
    radius = 1      # 和训练模型时保持一致
    atom_dict_path = r'atom_dict_174.csv'
    bond_dict_path = r'bond_dict_174.csv'
    fingerprint_dict_path = r'fingerprint_dict_174.csv'
    data_path = r'test_data_for_pre.csv'
    pre_save_path = r'test_pre_result.csv'

    """加载模型"""
    model = torch.load(r'model_174.pth', map_location=torch.device('cpu'))
    torch.no_grad()

    """加载带训练数据"""
    data = pd.read_csv(data_path).values
    data_smiles_D = data[:, 0]
    data_smiles_A = data[:, 4]
    HOMO_D = data[:, 2]
    LUMO_D = data[:, 3]
    HOMO_A = data[:, 6]
    LUMO_A = data[:, 7]
    descriptors = data[:, 8:]
    pre_results = []
    for i in range(len(data_smiles_D)):
        mol_D = Chem.AddHs(Chem.MolFromSmiles(data_smiles_D[i]))
        atoms_D = extract_atoms_nodes(mol_D, atom_dict_path)
        i_jbond_dict_D = extract_ijbonddict(mol_D, bond_dict_path)
        fingerprints_D = extract_fingerprint(radius, atoms_D, i_jbond_dict_D, fingerprint_dict_path)
        adjacency_D = Chem.GetAdjacencyMatrix(mol_D)

        mol_A = Chem.AddHs(Chem.MolFromSmiles(data_smiles_A[i]))
        atoms_A = extract_atoms_nodes(mol_A, atom_dict_path)
        i_jbond_dict_A = extract_ijbonddict(mol_A, bond_dict_path)
        fingerprints_A = extract_fingerprint(radius, atoms_A, i_jbond_dict_A, fingerprint_dict_path)
        adjacency_A = Chem.GetAdjacencyMatrix(mol_A)

        x = (fingerprints_D, HOMO_D[i], LUMO_D[i], adjacency_D, fingerprints_A, HOMO_A[i], LUMO_A[i], adjacency_A, descriptors[i, :])
        pre_result = model(x)
        pre_result_np = pre_result.detach().numpy()[0]
        print(pre_result_np)
        pre_results.append(pre_result_np)
    pre_results_pd = pd.DataFrame(pre_results)
    pre_results_pd.to_csv(pre_save_path)




