#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/08 6:58 오후
# @Author    : Junhyung Kwon
# @Site      : https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/master/attack/Attacker.py
# @File      : attacker.py
# @Software  : PyCharm
from abc import ABC

import torch


class Attacker(ABC):
    def __init__(self, model, config):
        """
        ## initialization ##
        :param model: Network to attack
        :param config : configuration to init the attack
        """
        self.config = config
        self.model = model
        self.clamp = (0, 1)

    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
        x = torch.clamp(x, *self.clamp)
        return x

    def __call__(self, x, y):
        x_adv = self.forward(x, y)
        return x_adv
