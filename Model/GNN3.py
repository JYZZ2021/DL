# 代码源自：华理小钊
# 时   间：2023/3/23 15:47
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        """
        dim指嵌入向量的维度，即用多少维来表示一个符号。此处表示用dim维服从N(0,1)分布的数据来表示N_fingerprints种的不同元素（词）
        最终会生成一个类似于词典之类的东西
        """
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(2 * dim + 61, 2 * dim + 61)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(2 * dim + 61, 1)

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors):
        sum_vectors = torch.sum(vectors, 0)
        return sum_vectors

    def gnn(self, inputs):

        """Cat或pad每个输入数据以进行批量处理。"""
        fingerprints_D, HOMO_D, LUMO_D, adjacency_D, fingerprints_A, HOMO_A, LUMO_A, adjacency_A, descriptors = inputs

        """GNN layer (update the fingerprint vectors)."""
        label_D = []
        for i in range(len(fingerprints_D)):
            if fingerprints_D[i] == 22222222:
                label_D.append(i)
            else:
                pass
        fingerprints_D = np.delete(fingerprints_D, label_D)

        label_A = []
        for i in range(len(fingerprints_A)):
            if fingerprints_A[i] == 22222222:
                label_A.append(i)
            else:
                pass
        fingerprints_A = np.delete(fingerprints_A, label_A)

        fingerprints_D = torch.tensor(fingerprints_D)
        adjacencies_D = torch.FloatTensor(adjacency_D)

        fingerprints_A = torch.tensor(fingerprints_A)
        adjacencies_A = torch.FloatTensor(adjacency_A)

        N_fingerprints_D = len(fingerprints_D)
        fingerprints_cat = torch.cat([fingerprints_D, fingerprints_A])

        fingerprint_vectors = self.embed_fingerprint(fingerprints_cat)

        fingerprint_vectors_D = fingerprint_vectors[:N_fingerprints_D]
        fingerprint_vectors_A = fingerprint_vectors[N_fingerprints_D:]

        fingerprint_vectors_np_D = fingerprint_vectors_D.detach().numpy()
        fingerprint_vectors_np_A = fingerprint_vectors_A.detach().numpy()

        zeros = np.zeros(54)        # 此处根据训练模型时的dim=50一致
        for j in label_D:
            fingerprint_vectors_np_D = np.insert(fingerprint_vectors_np_D, j, zeros, axis=0)
        fingerprint_vectors_D = torch.FloatTensor(fingerprint_vectors_np_D)
        # 此处表示依据词典对fingerprints中的每个元素（词）进行编码，最终生成一个len(fingerprints) * dim 的张量

        for m in label_A:           # 此处根据训练模型时的dim=50一致
            fingerprint_vectors_np_A = np.insert(fingerprint_vectors_np_A, m, zeros, axis=0)
        fingerprint_vectors_A = torch.FloatTensor(fingerprint_vectors_np_A)

        for l in range(4):          # 此处根据训练模型时的layer_hidden=2保持一致
            hs = self.update(adjacencies_D, fingerprint_vectors_D, l)
            fingerprint_vectors_D = F.normalize(hs, 2, 1)  # 标准化.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors_D = self.sum(fingerprint_vectors_D)

        for l in range(4):          # 此处根据训练模型时的layer_hidden=2保持一致
            hs = self.update(adjacencies_A, fingerprint_vectors_A, l)
            fingerprint_vectors_A = F.normalize(hs, 2, 1)  # 标准化.

        molecular_vectors_A = self.sum(fingerprint_vectors_A)

        HOMO_D = torch.tensor([HOMO_D])
        LUMO_D = torch.tensor([LUMO_D])
        HOMO_A = torch.tensor([HOMO_A])
        LUMO_A = torch.tensor([LUMO_A])
        descriptors = descriptors.astype(float)
        descriptors = torch.FloatTensor(descriptors)

        input_vectors = torch.cat([molecular_vectors_D, HOMO_D, LUMO_D,
                                   molecular_vectors_A, HOMO_A, LUMO_A, descriptors])

        return input_vectors

    def mlp(self, vectors):
        """Classifier or regressor based on multilayer perceptron."""
        for l in range(9):          # 此处根据训练模型时的layer_out = 9保持一致
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward(self, data):

        inputs = data

        molecular_vectors = self.gnn(inputs)
        predicted_values = self.mlp(molecular_vectors)
        return predicted_values
