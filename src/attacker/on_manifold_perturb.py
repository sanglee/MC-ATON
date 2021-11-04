#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/13 2:58 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : on_manifold_perturb.py
# @Software  : PyCharm

from abc import ABC

import torch
import torch.nn as nn
import numpy as np

class OnManifoldPerturb(ABC):
    def __init__(self, gan, num_step=20, batch_size=128, hidden_size=128):
        self.num_step = num_step
        self.gan = gan
        self.target_classifier = self.gan.discriminator
        self.encoder = self.gan.encoder
        self.decoder = self.gan.decoder
        pert = np.random.random((batch_size, hidden_size))
        self.zeta = nn.Parameter(torch.from_numpy(pert), requires_grad=True)

        self.target_classifier.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.gan.eval()


    def iter(self, X, y):
        x_adv = X.detach().clone()

        for X, y in range(self.num_step):

            self.encoder.zero_grad()
            self.decoder.zero_grad()
            self.target_classifier.zero_grad()

            mu, var = self.encoder(X)
            z = self.gan.reparameterize(mu, var)
            z = z + self.zeta
            X_tilda = self.decoder(z)

            delta = X_tilda - X



# class Linf_PGD(Attacker):
#     def __init__(self, model, config, target=None):
#         super(Linf_PGD, self).__init__(model, config)
#         self.target = target
#
#     def forward(self, x, y):
#         """
#         :param x: Inputs to perturb
#         :param y: Ground-truth label
#         :param target : Target label
#         :return adversarial image
#         """
#         x_adv = x.detach().clone()
#         if self.config['random_init']:
#             x_adv = self._random_init(x_adv)
#         for _ in range(self.config['attack_steps']):
#             x_adv.requires_grad = True
#             self.model.zero_grad()
#             logits = self.model(x_adv)  # f(T((x))
#             if self.target is None:
#                 # Untargeted attacks - gradient ascent
#
#                 loss = F.cross_entropy(logits, y, reduction="sum")
#                 loss.backward()
#                 grad = x_adv.grad.detach()
#                 grad = grad.sign()
#                 x_adv = x_adv + self.config['attack_lr'] * grad
#             else:
#                 # Targeted attacks - gradient descent
#                 assert self.target.size() == y.size()
#                 loss = F.cross_entropy(logits, self.target)
#                 loss.backward()
#                 grad = x_adv.grad.detach()
#                 grad = grad.sign()
#                 x_adv = x_adv - self.config['attack_lr'] * grad
#
#             # Projection
#             delta = torch.clamp(x_adv - x, min=-self.config['eps'], max=self.config['eps'])
#             x_adv = x + delta
#             x_adv = x_adv.detach()
#             x_adv = torch.clamp(x_adv, *self.clamp)
#
#         return x_adv, delta