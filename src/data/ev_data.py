import math
import os

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import torch
from scipy import io
from torch.utils import data


def make_ev_data(look_back, horizon_length, stride, horizon_stride, label_index=3):
    path = '/workspace/Dataset/TSData/Electric_Vehicle_data/'

    dfs = []
    for i in range(1, 6):
        file = 'route%d.mat' % i
        df = preprocess(path, file)
        dfs.append(df)

    # scaler fitting
    scaled_data, scaler = fit_scaler(dfs)

    train_X, train_y = [], []
    test_X, test_y = [], []

    for i, d in enumerate(scaled_data):
        if i == 0:
            test_X, test_y = dataToNumpy(d, look_back, horizon_length, stride, horizon_stride, label_index)
        else:
            tempX, tempy = dataToNumpy(d, look_back, horizon_length, stride, horizon_stride, label_index)
            if len(train_X) == 0:
                train_X, train_y = tempX, tempy
            else:
                train_X, train_y = np.concatenate((train_X, tempX)), np.concatenate((train_y, tempy))

    return (train_X, train_y), (test_X, test_y), scaler


def fit_scaler(data):
    # normalization
    scaler = preprocessing.StandardScaler()

    d = None
    for i in range(5):
        if i == 0:
            d = data[i]
        else:
            d = pd.concat((d, data[i]))
    d = scaler.fit(d)

    scaled_data = []
    for i in range(5):
        scaled_data.append(scaler.transform(data[i]))

    return scaled_data, scaler


def preprocess(path, file):
    # preprocessing
    mat_file = io.loadmat(os.path.join(path, file))
    df_data = {}
    for key in mat_file.keys():
        if not "__" in key:
            df_data[key] = mat_file[key].squeeze()

    df = pd.DataFrame(df_data)
    df = df.ffill().bfill()
    df.index = df['Time']
    df = df.drop('Time', axis=1)
    label = df.columns.to_numpy()

    return df


def dataToNumpy(data, look_back, horizon_length, window_stride, horizon_stride, label_index=3):
    total_data_size = math.trunc((data.shape[0] - look_back - horizon_length + 1) / window_stride)
    #     print(total_data_size)
    features = np.zeros((total_data_size, int(look_back / window_stride), 5))
    labels = np.zeros((total_data_size, int(horizon_length / horizon_stride)))
    j = 0
    while (j + look_back + horizon_length) < data.shape[0]:
        feature = np.array(data[j:(j + look_back)])
        label = np.array(data[(j + look_back):(j + look_back + horizon_length), label_index]).squeeze()
        label = label[::-1 * horizon_stride]
        label = np.flip(label)
        #         label = np.array(data[j+look_back+horizon_length, 3]).squeeze()
        #         label = np.digitize(label, bins = np.array([0, 0.25, 0.5, 0.75]), right=True)
        features[int(j / window_stride)] = feature
        labels[int(j / window_stride)] = label
        j += window_stride
    return features, labels


def get_loader(x, y, batch_size, shuffle=True, drop_last=True):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    #     print(x.shape, y.shape)
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


def ev_data_loader(lag=100, horizon=50, stride=1, horizon_stride=2, batch_size=100, label_index=3):
    # X, y, scaler = make_ev_data(100, 50, 1, label_index=3)
    trainset, testset, scaler = make_ev_data(lag, horizon, stride, horizon_stride, label_index)
    trainload = get_loader(trainset[0], trainset[1], batch_size)
    testload = get_loader(testset[0], testset[1], batch_size)

    return trainload, testload, scaler
