#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/08 6:59 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : pgd.py
# @Software  : PyCharm

import torch
import torch.nn.functional as F


class LinfPGDAttack(object):
    def __init__(self,
                 model,
                 epsilon=0.0314,
                 k=7,
                 alpha=0.00784):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x
