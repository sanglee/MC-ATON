from __future__ import print_function, division

import os
# Ignore warnings
import warnings

from numpy import isnan
from numpy import nan
from sklearn.model_selection import train_test_split

from utils import save_data

warnings.filterwarnings("ignore")

import torch
from torch.utils import data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_loader(x, y, batch_size, shuffle=True, drop_last=True):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    #     print(x.shape, y.shape)
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


class TSDataset:
    def __init__(self, file, root_dir, window_size, prediction_length, test_size, transform, dilation=60):
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.root_dir = root_dir
        self.file = file
        self.transform = transform
        self.test_size = test_size
        self.dilation = dilation

    def make_dataloader(self, type='train', batch_size=100):
        # if type == 'train':
        train_loader = get_loader(self.X_train, self.y_train, batch_size) # .reshape(self.X_train.shape[0], -1)
        # elif type == 'valid':
        val_loader = get_loader(self.X_valid, self.y_valid, batch_size) # .reshape(self.X_valid.shape[0], -1)
        # else:
        test_loader = get_loader(self.X_test, self.y_test, batch_size) # .reshape(self.X_test.shape[0], -1)

        return train_loader, val_loader, test_loader

    def __channel__(self):
        return self.data.shape[1]

    def __len__(self):
        """
        :return: data_size
        """
        return len(self.data)

    def save_scaler(self):
        save_data(self.scaler, '../data_', 'scaler.pkl')

    def _window_shifting(self, y_index):
        """

        :return:
             X_total_train,
             y_total_train,
             X_total_valid,
             y_total_valid,
             X_total_test,
             y_total_testd
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.data[:, y_index], test_size=self.test_size,
                                                            shuffle=False)

        validation_spot = int(len(X_train) * 0.9)
        test_perm = np.random.permutation(len(X_test) - (self.window_size + self.prediction_length) + 1)

        X_total_test = np.zeros(
            (X_test.shape[0] - (self.window_size + self.prediction_length) + 1, self.window_size, X_test.shape[1]))
        y_total_test = np.zeros(
            (X_test.shape[0] - (self.window_size + self.prediction_length) + 1, self.prediction_length))

        X_total_test, y_total_test = self._make_dataset(X_total_test, y_total_test, X_test, test_perm)

        _X_train = X_train[:validation_spot]
        _X_valid = X_train[validation_spot:]
        _y_train = y_train[:validation_spot]
        _y_valid = y_train[validation_spot:]

        perm_train = np.random.permutation(len(_X_train) - (self.window_size + self.prediction_length) + 1)
        perm_valid = np.random.permutation(len(_X_valid) - (self.window_size + self.prediction_length) + 1)

        X_total_train = np.zeros(
            (len(_X_train) - (self.window_size + self.prediction_length) + 1, self.window_size, _X_train.shape[1]))
        y_total_train = np.zeros(
            (len(_y_train) - (self.window_size + self.prediction_length) + 1, self.prediction_length))
        X_total_valid = np.zeros(
            (len(_X_valid) - (self.window_size + self.prediction_length) + 1, self.window_size, _X_train.shape[1]))
        y_total_valid = np.zeros(
            (len(_y_valid) - (self.window_size + self.prediction_length) + 1, self.prediction_length))

        X_total_train, y_total_train = self._make_dataset(X_total_train, y_total_train, _X_train, perm_train)
        X_total_valid, y_total_valid = self._make_dataset(X_total_valid, y_total_valid, _X_valid, perm_valid)

        return X_total_train, y_total_train, X_total_valid, y_total_valid, X_total_test, y_total_test

    def _make_dataset(self, X_total, y_total, data, perm):
        for j, idx in enumerate(perm):
            X = data[idx:idx + self.window_size]
            y = data[idx + self.window_size:idx + self.window_size + self.prediction_length][:, 4]
            X = np.transpose(X)
            # X_total[j] = X
            y_total[j] = y

        return X_total, y_total


class MetroTrafficDataset(TSDataset):
    def __init__(self, file, root_dir, window_size=24 * 7, prediction_length=24, test_size=0.2, transform=None):
        super().__init__(file, root_dir, window_size, prediction_length, test_size, transform, dilation=60)
        self.data, self.columns, self.scaler = self._prepare_data()
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self._window_shifting(
            y_index=4)

    def _prepare_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.file))

        holiday_one_hot = pd.get_dummies(df.holiday)
        weather_main_one_hot = pd.get_dummies(df.weather_main)
        weather_description_one_hot = pd.get_dummies(df.weather_description)

        numeric_data = df.drop(['holiday', 'weather_main', 'weather_description', 'date_time'], axis=1)

        numeric_columns = numeric_data.columns
        columns = np.concatenate((numeric_data.columns, holiday_one_hot.columns, weather_main_one_hot.columns,
                                  weather_description_one_hot.columns))

        scaler = StandardScaler()
        numeric_data = scaler.fit_transform(numeric_data)

        data = np.concatenate((numeric_data, holiday_one_hot, weather_main_one_hot, weather_description_one_hot),
                              axis=1)

        return data, columns, scaler


class HouseholdElectricDataset(TSDataset):
    def __init__(self, file, root_dir, window_size=24 * 7, prediction_length=24, test_size=0.2, dilation=60,
                 transform=None, y_index=0):
        super().__init__(file, root_dir, window_size, prediction_length, test_size, transform, dilation=dilation)
        self.data, self.columns, self.scaler = self._prepare_data()
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self._window_shifting(
            y_index)

    def _prepare_data(self):
        print(os.path.join(self.root_dir, self.file))
        df = pd.read_csv(os.path.join(self.root_dir, self.file), header=0, low_memory=False, infer_datetime_format=True,
                         index_col=['datetime'])
        df.index = pd.to_datetime(df.index)
        df = df.resample('%dMin' % self.dilation).mean()
        columns = df.columns
        scaler = StandardScaler()
        df = scaler.fit_transform(df)

        return df, columns, scaler

    def _fill_missing(self, values):
        one_day = 60 * 24
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                if isnan(values[row, col]):
                    values[row, col] = values[row - one_day, col]

    def preprocessing(self, file_name='household_power_consumption.csv'):
        d = os.path.join(self.root_dir, file_name)

        # load all data
        dataset = pd.read_csv(d, sep=';', header=0, low_memory=False, infer_datetime_format=True,
                              parse_dates={'datetime': [0, 1]}, index_col=['datetime'])

        # mark all missing values
        dataset.replace('?', nan, inplace=True)

        # make dataset numeric
        dataset = dataset.astype('float32')

        # fill missing
        self._fill_missing(dataset.values)

        values = dataset.values
        dataset['sub_metering_4'] = (values[:, 0] * 1000 / 60) - (values[:, 4] + values[:, 5] + values[:, 6])

        return dataset
