import pandas as pd
import os

import torch
from sklearn.model_selection import train_test_split


def ccfd(config):
    train_data = []
    test_data = []
    current = os.path.dirname(__file__)
    global_csv = pd.read_csv(current + "\\ccfd" + "\\ccfd_global.csv")
    train_csv = pd.read_csv(current + "\\ccfd" + "\\ccfd_train.csv")
    test_csv = pd.read_csv(current + "\\ccfd" + "\\ccfd_test.csv")
    data_agg_val = global_csv.iloc[:, -1]
    data_agg = global_csv.iloc[:, :-1]
    for key in config:
        train_data.append(train_csv.iloc[:, config[key]])
        test_data.append(test_csv.iloc[:, config[key]])
    train_data_val = train_csv.iloc[:, -1]
    test_data_val = test_csv.iloc[:, -1]
    for i in range(len(train_data)):
        train_data[i] = torch.FloatTensor(train_data[i].values)
        test_data[i] = torch.FloatTensor(test_data[i].values)
    data_agg = torch.FloatTensor(data_agg.values)
    data_agg_val = torch.FloatTensor(data_agg_val.values)
    train_data_val = torch.FloatTensor(train_data_val.values)
    test_data_val = torch.FloatTensor(test_data_val.values)
    return train_data, test_data, train_data_val, test_data_val, data_agg, data_agg_val
