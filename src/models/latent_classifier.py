#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/22 3:20 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : latent_classifier.py
# @Software  : PyCharm
import torch
import torch.nn as nn
from torch.autograd import Variable

class LatentClf(nn.Module):
    def __init__(self):
        super(LatentClf, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=2, stride=2)  # in_channels=3
        self.bn1 = nn.BatchNorm1d(64, momentum=0.9)
        self.maxpool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(128, momentum=0.9)
        self.fc1 = nn.Linear(128 * 16, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(256, momentum=0.9)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out.view(x.size(0),-1)
        out = self.relu(self.bn3(self.fc1(out)))
        out = self.relu(self.fc2(out))
        return out