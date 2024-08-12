import timeit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from rdkit import Chem
import os
import random


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, dropout):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        """
        dim指嵌入向量的维度，即用多少维来表示一个符号。此处表示用dim维服从N(0,1)分布的数据来表示N_fingerprints种的不同元素（词）
        最终会生成一个类似于词典之类的东西
        """
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(2 * dim + 59, 2 * dim + 59)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(2 * dim + 59, 1)
        self.dropout = nn.Dropout(dropout)


    def pad(self, matrices, pad_value):
        '''
        将输入的矩阵元组matrices中的各个tensor矩阵沿左上到右下的对角线拼接起来，其他位置填充pad_value（例如0）
        '''
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        hidden_vectors = self.dropout(hidden_vectors)
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        # torch.split(vectors, axis, dim=0) dim默认为0 表示在dim=0方向按照axis进行分区
        # torch.sum(v, 0)表示按列相加
        return torch.stack(sum_vectors)     # 默认按列拼接张量

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat或pad每个输入数据以进行批量处理。"""
        fingerprints_D, adjacencies_D, molecular_size_D, HOMO_D, LUMO_D, \
            fingerprints_A, adjacencies_A, molecular_size_A, HOMO_A, LUMO_A, descriptors = inputs
        fingerprints_D = torch.cat(fingerprints_D)      # 把每个分子结构的指纹整合成一维张量
        fingerprints_A = torch.cat(fingerprints_A)
        adjacencies_D = self.pad(adjacencies_D, 0)
        adjacencies_A = self.pad(adjacencies_A, 0)

        N_fingerprints_D = len(fingerprints_D)
        fingerprints_cat = torch.cat([fingerprints_D, fingerprints_A])

        fingerprint_vectors = self.embed_fingerprint(fingerprints_cat)
        # 此处表示依据词典对fingerprints中的每个元素（词）进行编码，最终生成一个len(fingerprints) * dim 的张量

        fingerprint_vectors_D = fingerprint_vectors[:N_fingerprints_D]
        fingerprint_vectors_A = fingerprint_vectors[N_fingerprints_D:]

        for l in range(layer_hidden):
            hs = self.update(adjacencies_D, fingerprint_vectors_D, l)
            fingerprint_vectors_D = F.normalize(hs, 2, 1)  # 标准化.

        molecular_vectors_D = self.sum(fingerprint_vectors_D, molecular_size_D)

        for l in range(layer_hidden):
            hs = self.update(adjacencies_A, fingerprint_vectors_A, l)
            fingerprint_vectors_A = F.normalize(hs, 2, 1)  # 标准化.

        molecular_vectors_A = self.sum(fingerprint_vectors_A, molecular_size_A)

        N = len(HOMO_D)

        HOMO_D = torch.cat(HOMO_D)
        LUMO_D = torch.cat(LUMO_D)
        HOMO_A = torch.cat(HOMO_A)
        LUMO_A = torch.cat(LUMO_A)
        descriptors = torch.cat(descriptors).reshape(N, 55)

        input_vectors = torch.cat([molecular_vectors_D, HOMO_D, LUMO_D,
                                   molecular_vectors_A, HOMO_A, LUMO_A, descriptors], 1)

        return input_vectors

    def mlp(self, vectors):
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
            vectors = self.dropout(vectors)
        outputs = self.W_property(vectors)
        return outputs

    def forward(self, data_batch, train=True):

        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            input_vectors = self.gnn(inputs)

            predicted_values = self.mlp(input_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        train_loss = loss_total / N
        return train_loss

class Validate(object):
    def __init__(self, model):
        self.model = model

    def val(self, dataset):
        model.eval()        # 切换模型为预测模型
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_val):
            data_batch = list(zip(*dataset[i:i+batch_val]))
            loss = self.model.forward(data_batch, train=True)
            loss_total += loss.item()
        val_loss = loss_total / N
        return val_loss

class Tester(object):
    def __init__(self, model):
        self.model = model


    def test_regressor(self, dataset):
        model.eval()
        N = len(dataset)
        sae = 0  # sum absolute error.
        sse = 0  # 剩余平方和
        sst = 0  # 总离差平方和
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_values, correct_values = self.model.forward(
                                               data_batch, train=False)
            correct_values_mean = np.mean(correct_values)
            sae += sum(np.abs(predicted_values-correct_values))
            sse += sum((np.abs(predicted_values - correct_values))**2)
            sst += sum((np.abs(correct_values - correct_values_mean))**2)
        mae = sae / N  # mean absolute error.
        rmse = (sse / N)**0.5
        r2 = 1-sse / sst
        return mae, rmse, r2

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

class EarlyStopping:
    """如果在给定的步长次数（patience）上验证集的损失没有显著下降，则提前停止训练"""
    def __init__(self, save_model_path, patience=20, verbose=False, delta=0):
        """
        参数:
            save_model_path : 模型保存文件夹
            patience (int): 上次验证集损失改善后多少次没有改善.
                            默认: 7
            verbose (bool): 如果为True，则为每个验证损失改进打印一条消息
                            默认: False
            delta (float): 监测数量的最小变化，以确定为改进。
                            默认: 0
        """
        self.save_model_path = save_model_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''当验证集损失减少时保存模型'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), save_model_path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def create_atoms(mol, atom_dict):
    """
    获取分子中的原子以及芳香性原子
    返回数组，如果原子为芳香性原子则对应的元素为元组，如乙酸甲酯C(=O)OC生成的结果如下所示
    array([0, 0, 1, 1, 0])
    其中0代表C，1代表O
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """
    获取分子中两个原子之间的键
    返回字典，字典的键为一个原子，值为一个元组，元组内为与之相连的另外一个原子和它们之间的键，如乙酸甲酯CC(=O)OC生成的结果如下所示，
    其中字典键值中的数字和元组第一个位置的数字均代表几号原子，元组第二个位置的数字0代表单键，1代表双键
    defaultdict(<function __main__.<lambda>()>,
            {0: [(1, 0)],
             1: [(0, 0), (2, 1), (3, 0)],
             2: [(1, 1)],
             3: [(1, 0), (4, 0)],
             4: [(3, 0)]})
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """
    从分子中提取指纹（通过Weisfeiler-Lehman（WL-test）算法不断聚合邻居信息，得到节点的新表示（向量），根据此新表示判断两个结构是否是同构体，
    如果得到得新向量相同，则不排除两个结构是同构体）
    根据原子的相邻原子的数量和种类以及彼此之间的键接方式更新原子ID（即更新atoms）得到新的array([*，*，*，。。。])
    """
    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):
            """
            考虑到每个节点的相邻节点和边，更新每个节点ID。
            更新后的节点ID是指纹ID。
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """
            考虑边两侧连接原子的种类，从而更新边。
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def split_dataset(dataset, ratio, seed):
    """打乱并划分数据集"""
    np.random.seed(seed)  # 划分时固定种子
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(radius, device, data_path, data_train_shuffle_path, data_test_shuffle_path, data_val_shuffle_path,
                    atom_dict_path, bond_dict_path, fingerprint_dict_path, seed, ratio):
    """加载数据集"""
    dataset = pd.read_csv(data_path).values

    """初始化x_dict，其中每个键是一种符号类型（例如原子和化学键），每个值是其索引"""
    atom_dict = defaultdict(lambda: len(atom_dict))
    """
    记录原子种类的映射
    如：
    defaultdict(<function __main__.<lambda>()>,
            {'*': 0,
             'C': 1,
             'O': 2,
             ('C', 'aromatic'): 3,
             'H': 4,
             ('N', 'aromatic'): 5,
             ('O', 'aromatic'): 6})
    """

    bond_dict = defaultdict(lambda: len(bond_dict))
    """
    记录键的种类的映射
    如：
    defaultdict(<function __main__.<lambda>()>,
            {'SINGLE': 0, 'AROMATIC': 1, 'DOUBLE': 2})
    """
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    """
    记录各个原子的ID（根据与其相连的原子数量和种类以及之间的键的种类分类）‘
    其长度表示分子结构种有多少种不同连接方式的原子
    如：
    defaultdict(<function __main__.<lambda>()>,
            {(0, ((1, 0),)): 0,
             (1, ((0, 0), (1, 0), (4, 0), (4, 0))): 1,
             (1, ((1, 0), (2, 0), (4, 0), (4, 0))): 2,
             ... ...
             (3, ((1, 0), (5, 1), (6, 1))): 25,
             (3, ((0, 0), (5, 1), (6, 1))): 26,
             (6, ((3, 1), (3, 1))): 27,
             (3, ((3, 1), (3, 1), (6, 1))): 28})
    第一行表示映射为0的原子（*）与映射为1的原子(C)以映射为0的键('SINGLE')连接，此类原子的ID为0
    """
    edge_dict = defaultdict(lambda: len(edge_dict))
    """
    记录各个键的ID（根据键的种类以及其两端连的原子种类和数量分类）
    其长度表示分子结构中有多少种键
    如：
    defaultdict(<function __main__.<lambda>()>,
            {((0, 1), 0): 0,
             ((1, 1), 0): 1,
             ((1, 4), 0): 2,
             ((1, 2), 0): 3,
             ((2, 3), 0): 4,
             ((3, 3), 1): 5,
             ((3, 4), 0): 6,
             ((1, 3), 0): 7,
             ((1, 2), 2): 8,
             ((0, 2), 0): 9,
             ((0, 3), 0): 10,
             ((3, 3), 0): 11,
             ((3, 5), 1): 12,
             ((3, 6), 1): 13})
    第一行表示两端连接：映射为0的原子（*）和映射为1的原子（C）且键的类型为映射为0（'SINGLE'） 的键的ID为0
    """

    def extract_inf_from_structure(dataset):
        dataset_ = []

        """提取数据集中每个分子结构信息，并转化为张量后存储在元组中，汇总到列表中"""
        for data in dataset:
            smiles_D = data[1]
            smiles_A = data[9]
            HOMO_D = data[3]
            LUMO_D = data[4]
            HOMO_A = data[11]
            LUMO_A = data[12]
            descriptors = data[22:]
            property = data[13]

            """使用上述定义的函数提取分子结构信息"""
            mol_D = Chem.AddHs(Chem.MolFromSmiles(smiles_D))
            mol_A = Chem.AddHs(Chem.MolFromSmiles(smiles_A))
            atoms_D = create_atoms(mol_D, atom_dict)
            atoms_A = create_atoms(mol_A, atom_dict)
            molecular_size_D = len(atoms_D)
            molecular_size_A = len(atoms_A)
            i_jbond_dict_D = create_ijbonddict(mol_D, bond_dict)
            i_jbond_dict_A = create_ijbonddict(mol_A, bond_dict)
            fingerprints_D = extract_fingerprints(radius, atoms_D, i_jbond_dict_D,
                                                fingerprint_dict, edge_dict)
            fingerprints_A = extract_fingerprints(radius, atoms_A, i_jbond_dict_A,
                                                  fingerprint_dict, edge_dict)
            adjacency_D = Chem.GetAdjacencyMatrix(mol_D)
            adjacency_A = Chem.GetAdjacencyMatrix(mol_A)

            """将上面的每个numpy数据转换为设备（即CPU或GPU）上的pytorch张量"""
            fingerprints_D = torch.LongTensor(fingerprints_D).to(device)
            fingerprints_A = torch.LongTensor(fingerprints_A).to(device)
            adjacency_D = torch.FloatTensor(adjacency_D).to(device)
            adjacency_A = torch.FloatTensor(adjacency_A).to(device)
            HOMO_D  = torch.FloatTensor([[float(HOMO_D)]]).to(device)
            LUMO_D  = torch.FloatTensor([[float(LUMO_D)]]).to(device)
            HOMO_A  = torch.FloatTensor([[float(HOMO_A)]]).to(device)
            LUMO_A  = torch.FloatTensor([[float(LUMO_A)]]).to(device)
            descriptors = descriptors.astype(float)
            descriptors = torch.FloatTensor(descriptors).to(device)
            property = torch.FloatTensor([[float(property)]]).to(device)

            dataset_.append((fingerprints_D, adjacency_D, molecular_size_D, HOMO_D, LUMO_D,
                             fingerprints_A, adjacency_A, molecular_size_A, HOMO_A, LUMO_A, descriptors, property))
        return dataset_

    """对数据集进行划分并保存数据集的划分方式"""
    dataset_train, dataset_test = split_dataset(dataset, ratio, seed)
    dataset_train, dataset_val = split_dataset(dataset_train, ratio, seed)
    dataset_train_pd = pd.DataFrame(dataset_train)
    dataset_test_pd = pd.DataFrame(dataset_test)
    dataset_val_pd = pd.DataFrame(dataset_val)
    dataset_train_pd.to_csv(data_train_shuffle_path)
    dataset_test_pd.to_csv(data_test_shuffle_path)
    dataset_val_pd.to_csv(data_val_shuffle_path)

    dataset_train = extract_inf_from_structure(dataset_train)
    dataset_test = extract_inf_from_structure(dataset_test)
    dataset_val = extract_inf_from_structure(dataset_val)
    N_fingerprints = len(fingerprint_dict)

    """保存原子字典、键字典和指纹字典"""
    atom_dict_pd = pd.DataFrame(atom_dict, index=[0])
    bond_dict_pd = pd.DataFrame(bond_dict, index=[0])
    fingerprint_dict_pd = pd.DataFrame(fingerprint_dict, index=[0])
    atom_dict_pd.to_csv(atom_dict_path)
    bond_dict_pd.to_csv(bond_dict_path)
    fingerprint_dict_pd.to_csv(fingerprint_dict_path)

    return dataset_train, dataset_test, dataset_val, N_fingerprints

if __name__ == "__main__":

    '''模型参数'''
    train_numbers = 500    # 训练次数
    seed = 1234             # 随机数种子
    radius = 1              # 执行 Weisfeiler-Lehman（WL-test）算法的次数
    ratio = 0.9             # 训练集占比
    batch_train = 32        # 训练集的批量大小
    batch_test = 32         # 测试集的批量大小
    batch_val = 32          # 验证集得批量大小
    decay_interval = 10     # 每迭代训练decay_interval次后更新学习率
    iteration = 1000        # 迭代次数
    lr_decay = 0.977        # 学习衰减率
    dropout = 0.1           # dropout损失概率

    '''文件路径'''
    property = 'PCE'                                       # 模型针对的性能
    save_path = '/home/zhangsz233/OPV-ALL-4/model/PCE/Input-DA_Stru+HOMO+LUMO+des/output/'        # 模型、文件保存路径
    data_path = '/home/zhangsz233/OPV-ALL-4/model/PCE/Input-DA_Stru+HOMO+LUMO+des/data_original+descriptors.csv'   # 数据路径

    '''指定GPU'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('模型训练使用GPU!')
    else:
        device = torch.device('cpu')
        print('模型训练使用CPU...')

    for i in range(train_numbers):

        '''数据集划分种子+1'''
        seed = seed + 1

        '''随机生成模型超参数'''
        dim = random.randint(50, 100)           # dim指嵌入向量的维度，即用多少维来表示一个符号（词）。
        layer_hidden = random.randint(2, 10)    # layer_hidden指图神经网络（MLP网络之前的网络）隐藏层的个数
        layer_output = random.randint(2, 10)    # MLP隐藏层的个数
        lr = random.uniform(0.002, 0.0005)       # 学习率

        '''文件保存路径'''
        data_train_shuffle_path = save_path + r'dataset_train_shuffle_' + str(i) + '.csv'   # 划分的训练集保存路径
        data_test_shuffle_path = save_path + r'dataset_test_shuffle_' + str(i) + '.csv'     # 划分的测试集保存路径
        data_val_shuffle_path = save_path + r'dataset_val_shuffle_' + str(i) + '.csv'
        save_model_path = save_path + r'model_' + str(i) + '.pth'                           # 模型保存路径
        atom_dict_path = save_path + r'atom_dict_' + str(i) + '.csv'                        # 原子字典保存路径
        bond_dict_path = save_path + r'bond_dict_' + str(i) + '.csv'                        # 键的字典保存路径
        fingerprint_dict_path = save_path + r'fingerprint_dict_' + str(i) + '.csv'          # 指纹字典保存路径
        file_result = save_path + 'result_' + str(i) + property + '.txt'                    # 训练过程记录及结果保存路径
        file_result_all = save_path + '_result_all' + '.txt'

        print('-'*100)
        print('预处理数据集')
        print('处理中......')
        (dataset_train, dataset_test, dataset_val, N_fingerprints) = create_datasets(radius, device, data_path, data_train_shuffle_path,
                                                                        data_test_shuffle_path, data_val_shuffle_path,
                                                                        atom_dict_path, bond_dict_path,
                                                                        fingerprint_dict_path, seed, ratio)
        print('-' * 100)
        print('模型参数:')
        print(f'radius:{radius}')
        print(f'dim:{dim}')
        print(f'layer_hidden:{layer_hidden}')
        print(f'layer_output:{layer_output}')
        print(f'train_set_ratio:{ratio}')
        print(f'batch_train:{batch_train}')
        print(f'batch_test:{batch_test}')
        print(f'lr:{lr}')
        print(f'lr_decay:{lr_decay}')
        print(f'decay_interval:{decay_interval}')
        print(f'iteration:{iteration}')
        print('-' * 100)
        print('预处理结束!')
        print('训练集大小:', len(dataset_train))
        print('测试集大小:', len(dataset_test))
        print('-'*100)
        print('训练模型:')
        model = MolecularGraphNeuralNetwork(N_fingerprints, dim, layer_hidden, layer_output, dropout).to(device)
        trainer = Trainer(model)
        tester = Tester(model)
        validate = Validate(model)
        print('模型参数数量:', sum([np.prod(p.size()) for p in model.parameters()]))
        print('-'*100)
        result = 'Epoch\tTime(sec)\tLoss_train\tLoss_val\tMAE_train\tMAE_test\tMAE_val\tRMSE_train\tRMSE_test\tRMSE_val' \
                 '\tR2_train\tR2test\tR2val'
        with open(file_result, 'w') as f:
            f.write(f'radius:{radius}' + '\n')
            f.write(f'dim:{dim}' + '\n')
            f.write(f'layer_hidden:{layer_hidden}' + '\n')
            f.write(f'layer_output:{layer_output}' + '\n')
            f.write(f'dropout:{dropout}' + '\n')
            f.write(f'train_set_ratio:{ratio}' + '\n')
            f.write(f'batch_train:{batch_train}' + '\n')
            f.write(f'batch_test:{batch_test}' + '\n')
            f.write(f'batch_val:{batch_val}' + '\n')
            f.write(f'lr:{lr}' + '\n')
            f.write(f'lr_decay:{lr_decay}' + '\n')
            f.write(f'decay_interval:{decay_interval}' + '\n')
            f.write(f'iteration:{iteration}' + '\n')
            f.write(f'训练集大小:{len(dataset_train)}' + '\n')
            f.write(f'测试集大小:{len(dataset_test)}' + '\n')
            f.write(f'验证集大小:{len(dataset_val)}' + '\n')
            f.write(f'模型参数数量:{sum([np.prod(p.size()) for p in model.parameters()])}' + '\n')
            f.write(result + '\n')
        print('开始训练......')
        print('训练结果每个epoch都保存在输出目录中!')
        np.random.seed(seed)
        start = timeit.default_timer()
        early_stopping = EarlyStopping(save_model_path)     # 提前停止

        for epoch in range(iteration):
            epoch += 1
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay     # 衰减学习率
            loss_train = trainer.train(dataset_train)
            loss_val = validate.val(dataset_val)

            train_mae, train_RMSE, train_R2 = tester.test_regressor(dataset_train)
            test_mae, test_RMSE, test_R2 = tester.test_regressor(dataset_test)
            val_mae, val_RMSE, val_R2 = tester.test_regressor(dataset_val)

            time = timeit.default_timer() - start

            if epoch == 1:
                minutes = time * iteration / 60
                hours = int(minutes / 60)
                minutes = int(minutes - 60 * hours)
                print('The training will finish in about', hours, 'hours', minutes, 'minutes.')
                print('-'*100)
                print(result)

            result = '\t'.join(map(str, [epoch, time, loss_train, loss_val, train_mae, test_mae, val_mae,
                                         train_RMSE, test_RMSE, val_RMSE, train_R2, test_R2, val_R2]))
            tester.save_result(result, file_result)
            print(result)

            '''提早停止'''
            early_stopping(loss_val, model)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练

        '''保存模型'''
        torch.save(model, save_model_path)

        # 将预测结果实时写入txt文件
        train_mae, train_RMSE, train_R2 = tester.test_regressor(dataset_train)
        test_mae, test_RMSE, test_R2 = tester.test_regressor(dataset_test)
        val_mae, val_RMSE, val_R2 = tester.test_regressor(dataset_val)

        if i == 0:
            with open(file_result_all, 'a+') as f:
                f.write(f'Number' + '\t')
                f.write(f'Train_R2' + '\t')
                f.write(f'Test_R2' + '\t')
                f.write(f'Val_R2' + '\t')
                f.write(f'Train_MAE' + '\t')
                f.write(f'Test_MAE' + '\t')
                f.write(f'Val_MAE' + '\n')

        with open(file_result_all, 'a+') as f:
            f.write(f'{i}' + '\t')
            f.write(f'{train_R2}' + '\t')
            f.write(f'{test_R2}' + '\t')
            f.write(f'{val_R2}' + '\t')
            f.write(f'{train_mae}' + '\t')
            f.write(f'{test_mae}' + '\t')
            f.write(f'{val_mae}' + '\n')