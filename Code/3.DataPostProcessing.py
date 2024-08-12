import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pickle

def main():
    data_all = pd.read_csv('descriptors_A_filter.csv', low_memory=True)

    # 定位Descripotors
    X = data_all.values[:, 0:]

    pipe = Pipeline([('min_max', MinMaxScaler()), ('PCA', PCA(n_components=0.95)), ('Transformer', PowerTransformer())])
    pipe.fit(X)
    X_New = pipe.transform(X)

    # 保存降维后的X
    data_pd = pd.DataFrame(X_New)

    # 保存文件
    data_pd.to_csv('PCA_A.csv', index=False)

    # 保存训练的转换器
    with open('PCA_A', 'wb') as fw:
        pickle.dump(pipe, fw)


if __name__ == '__main__':
    main()
