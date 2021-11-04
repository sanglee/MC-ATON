#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/14 2:46 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : on_manifold_generation.py
# @Software  : PyCharm
import torch
import torch.nn.functional as F

class OnManifoldPerturbation(object):
    def __init__(self, classifier, gan, device, eta=0.3, k=7, alpha=0.0784):
        self.classifier = classifier
        self.device = device
        self.gan = gan
        self.encoder = gan.encoder
        self.decoder = gan.decoder
        self.eta = eta
        self.k = k
        self.alpha = alpha

    def perturb(self, X, y):
        mu, var = self.encoder(X.to(self.device))
        z = self.gan.reparameterize(mu, var)
        zeta = z.detach().clone()

        for i in range(self.k):
            zeta.requires_grad_()
            with torch.enable_grad():
                x_tilde_pert = self.decoder(z + zeta)
                pert_logits = self.classifier(x_tilde_pert)
                loss = F.cross_entropy(pert_logits, y.to(self.device), reduction="sum")
            grad = torch.autograd.grad(loss, [zeta])[0]
            z_pert = z.detach() + self.alpha * torch.sign(grad.detach())
            zeta = torch.clamp(z_pert - z, min=-self.eta, max=self.eta)
            z_pert = torch.clamp(z + zeta, min=-2., max=2.)
            x = self.decoder(z_pert)
            x = torch.clamp(x, 0, 1)

        return z, z_pert, x


class OnManifoldPerturbation_v2(object):
    def __init__(self, classifier, vae, device, eta=0.3, k=7, alpha=0.0784):
        self.classifier = classifier
        self.device = device
        self.vae = vae
        self.eta = eta
        self.k = k
        self.alpha = alpha

    def perturb(self, X, y):
        z, mu, var = self.vae.encode(X.to(self.device))
        zeta = z.detach().clone()

        for i in range(self.k):
            zeta.requires_grad_()
            with torch.enable_grad():
                x_tilde_pert = self.vae.decode(z + zeta)
                pert_logits = self.classifier(x_tilde_pert)
                loss = F.cross_entropy(pert_logits, y.to(self.device), reduction="sum")
            grad = torch.autograd.grad(loss, [zeta])[0]
            z_pert = z.detach() + self.alpha * torch.sign(grad.detach())
            zeta = torch.clamp(z_pert - z, min=-self.eta, max=self.eta)
            z_pert = torch.clamp(z + zeta, min=-2., max=2.)
            x = self.vae.decode(z_pert)
            x = torch.clamp(x, 0, 1)

        return z, z_pert, x
