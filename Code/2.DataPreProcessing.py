#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np


# 1.去除计算报错的列
# 2.去除全零的列

if __name__ == '__main__':

    # 读取数据
    data_pd = pd.read_csv('descriptors_A.csv', low_memory=False)

    # 获得列名列表
    column_list = [column for column in data_pd]

    # 转成numpy array
    data_np = data_pd.values
    # print(data_np)
    m, n = data_np.shape
    # print(m, n)

    # 遍历[0:]每一列(0,1,2为smiles,eib,bandgap_chain).如果该列中存在不是0的值,则删除;如果全是0,则删除
    filter_descriptors = {}
    for i in range(0, n):
        tag = 0
        sum = 0

        # 获取列名(描述符名字)和该列所有值
        des_Name = column_list[i]
        des_value_column = data_np[:, i].tolist()

        # 该两列为True or False,不考虑
        if des_Name == 'GhoseFilter' or des_Name == 'Lipinski':
            continue

        # 遍历该描述符所有值，进行处理
        for value in des_value_column:
            if isinstance(value, str):
                tag = 1
                break
            if np.isnan(value):
                tag = 1
                break
            sum += abs(value)
        if tag == 1:
            continue
        if sum == 0:
            continue

        # 保存满足条件的描述符列
        filter_descriptors[des_Name] = des_value_column

    filter_descriptors_pd = pd.DataFrame(filter_descriptors)

    # 保存文件
    filter_descriptors_pd.to_csv('descriptors_A_filter.csv', index=False)
