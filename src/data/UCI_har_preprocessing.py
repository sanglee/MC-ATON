#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/10/28 13:30
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : UCI_har_preprocessing.py
# @Software  : PyCharm

import pandas as pd
import json
import numpy as np
from collections import Counter
import os
from torch.utils import data
import torch

def get_loader(x, y, batch_size, shuffle=True, drop_last=True):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    #     print(x.shape, y.shape)
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


def load_x(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        with open(signal_type_path, "r") as f:
            X_signals.append(
                [np.array(serie, dtype=np.float32)
                 for serie in [row.replace('  ', ' ').strip().split(' ') for row in f]]
            )

    return np.transpose(X_signals, (1, 0, 2))


def load_y(y_path):
    # Read dataset from disk, dealing with text file's syntax
    with open(y_path, "r") as f:
        y = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in f
            ]],
            dtype=np.int32
        )

    y = y.reshape(-1, )
    # Substract 1 to each output class for friendly 0-based indexing
    return y - 1

def uci_preprocessing(data_path = "./uci_data/UCI_HAR_Dataset/", save_path='./uci_data/np', input_signal_type=None):
    DATASET_PATH = data_path

    if input_signal_type == None:
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]
    else:
        INPUT_SIGNAL_TYPES = input_signal_type



    train_x_signals_paths = [
        DATASET_PATH + "train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    test_x_signals_paths = [
        DATASET_PATH + "test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    ]

    train_y_path = DATASET_PATH + "train/y_train.txt"
    test_y_path = DATASET_PATH + "test/y_test.txt"

    train_x = load_x(train_x_signals_paths)
    test_x = load_x(test_x_signals_paths)

    train_y = load_y(train_y_path)
    test_y = load_y(test_y_path)
    # train_y_matrix = np.asarray(pd.get_dummies(train_y), dtype=np.int8)
    # test_y_matrix = np.asarray(pd.get_dummies(test_y), dtype=np.int8)

    print(train_y, Counter(train_y))
    print(test_y, Counter(test_y))
    # print(train_y)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    np.save(os.path.join(save_path, "np_train_x.npy"), train_x)
    np.save(os.path.join(save_path, "np_train_y.npy"), train_y)
    np.save(os.path.join(save_path, "np_test_x.npy"), test_x)
    np.save(os.path.join(save_path, "np_test_y.npy"), test_y)

def uci_har(data_base_path='UCI_data/np/'):
    train_x = np.load(os.path.join(data_base_path, 'np_train_x.npy'))
    train_y = np.load(os.path.join(data_base_path, 'np_train_y.npy'))
    test_x = np.load(os.path.join(data_base_path, 'np_test_x.npy'))
    test_y = np.load(os.path.join(data_base_path, 'np_test_y.npy'))

    train_loader = get_loader(train_x, train_y, 100, drop_last=False)
    test_loader = get_loader(test_x, test_y, 100, shuffle=False, drop_last=False)

    return train_loader, test_loader

