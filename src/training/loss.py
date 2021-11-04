#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/02 1:32 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : loss.py
# @Software  : PyCharm

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class RobustLoss(_Loss):
    def __init__(self, gamma, dist="mse_loss"):
        """
        Objective for robust regression model
        :param gamma: parameter for loss term between mse loss and loss between pert and benign
        :param dist: base distance term (mse_loss, l1_loss)
        """
        super(RobustLoss, self).__init__()
        assert 0 <= gamma <= 1, "Gamma must between (0, 1)"
        assert hasattr(F, dist), "Dist not in torch.nn.functional"
        self.gamma = gamma
        self.dist = dist

    def _robust_loss(self, benign, pert, target):
        seq_term = torch.arange(benign.shape[1], dtype=torch.float32)
        distance = getattr(F, self.dist)(benign, target)
        tilde_distance = getattr(F, self.dist)(pert, target)

        loss = self.gamma * distance + (1 - self.gamma) * tilde_distance
        return loss, distance, tilde_distance

    def forward(self, benign, pert, target):
        loss, distance, tilde_distance = self._robust_loss(benign, pert, target)
        return loss, distance, tilde_distance
