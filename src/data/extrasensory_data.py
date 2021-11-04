#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/10/27 16:20
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : extrasensory_data.py
# @Software  : PyCharm

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import gzip
from io import StringIO


def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')

    # The first column should be timestamp:
    assert columns[0] == 'timestamp'
    # The last column should be label_source:
    assert columns[-1] == 'label_source'

    # Search for the column of the first label:
    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break
        pass

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:')
        label_names[li] = label.replace('label:', '')
        pass

    return (feature_names, label_names)


def parse_body_of_csv(csv_str, n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO.StringIO(csv_str), delimiter=',', skiprows=1)

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int)

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)]

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1]  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.  # Y is the label matrix

    return (X, Y, M, timestamps)


'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''


def read_user_data(uuid):
    user_data_file = '%s.features_labels.csv.gz' % uuid

    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rb') as fid:
        csv_str = fid.read()
        pass

    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return (X, Y, M, timestamps, feature_names, label_names)
