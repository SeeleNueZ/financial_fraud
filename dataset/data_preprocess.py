import pandas as pd
import os

import torch
from sklearn.model_selection import train_test_split


def ccfd(config):
    current =  os.path.dirname(__file__)
    data = pd.read_csv(current + "\\post" + "\\ccfd_slice.csv")
    # print(data)
    # 分离label
    data_label = data.iloc[:, 28]
    data = data.iloc[:, 0:28]
    # 完成标准化
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std
    # 尝试归一化
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    data = (data - min_val) / (max_val - min_val + 1e-8)  # 避免除零
    # print(data)

    # 分离聚合端数据
    data = pd.concat([data, data_label], axis=1)
    list_out = train_test_split(data, test_size=0.16, shuffle=True)
    # print(list_out)
    data_agg = list_out[1]
    data_agg_val = data_agg.iloc[:, -1]
    data_agg = data_agg.iloc[:, :-1]
    # 分离测试集训练集
    client_data = train_test_split(list_out[0], test_size=0.17, shuffle=True)
    print(client_data)
    train_data = []
    test_data = []
    for key in config:
        train_data.append(client_data[0].iloc[:, config[key]])
        test_data.append(client_data[1].iloc[:, config[key]])
    train_data_val = client_data[0].iloc[:, -1]
    test_data_val = client_data[1].iloc[:, -1]
    for i in range(len(train_data)):
        train_data[i] = torch.FloatTensor(train_data[i].values)
        test_data[i] = torch.FloatTensor(test_data[i].values)
    data_agg = torch.FloatTensor(data_agg.values)
    data_agg_val = torch.FloatTensor(data_agg_val.values)
    train_data_val = torch.FloatTensor(train_data_val.values)
    test_data_val = torch.FloatTensor(test_data_val.values)
    # print(train_data)
    print(data_agg.shape)
    print(data_agg_val.shape)
    return train_data, test_data, train_data_val, test_data_val, data_agg, data_agg_val


if __name__ == "__main__":
    c1 = [x for x in range(0, 6)]
    c2 = [x for x in range(6, 10)]
    c3 = [x for x in range(10, 20)]
    c4 = [x for x in range(20, 28)]
    config = {0: c1, 1: c2, 2: c3, 3: c4}
    ccfd(config=config)
