#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/10/28 14:25
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : har_clf.py
# @Software  : PyCharm

import torch
from torch import nn

class HARClassifier(nn.Module):
    def __init__(self):
        super(HARClassifier, self).__init__()

        self.conv_layers =  nn.Sequential(
            nn.Conv1d(9, 32, 3, 1, 1),
            nn.MaxPool1d(4, 2, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.MaxPool1d(4, 2, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.MaxPool1d(4, 2, 1),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 6),
        )

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
