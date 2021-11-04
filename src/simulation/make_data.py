#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/18 10:51 오전
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : make_data.py
# @Software  : PyCharm

# Needed for plotting
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from attacker import LinfPGDAttack
from training import structured_prune
from utils import AverageMeter

rand_state = 11


def accuracy(net, test_loader, device):
    net.eval()
    benign_correct = 0
    total_set = 0
    for X, y in test_loader:
        X = X.to(device)
        out = net(X)
        out = F.softmax(out, dim=1)

        _, predicted = out.max(1)
        benign_correct += predicted.eq(y.to(device)).sum().item()
        total_set += X.size(0)

    benign_acc = str(benign_correct / total_set)
    print('accuracy: {0}'.format(benign_acc))
    return benign_acc


def concatenate(array1, array2, axis=0):
    """
    Basically a wrapper for numpy.concatenate, with the exception
    that the array itself is returned if its None or evaluates to False.
    :param array1: input array or None
    :type array1: mixed
    :param array2: input array
    :type array2: numpy.ndarray
    :param axis: axis to concatenate
    :type axis: int
    :return: concatenated array
    :rtype: numpy.ndarray
    """

    assert isinstance(array2, np.ndarray)
    if array1 is not None:
        assert isinstance(array1, np.ndarray)
        return np.concatenate((array1, array2), axis=axis)
    else:
        return array2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Tilda(torch.nn.Module):
    def __init__(self):
        super(Tilda, self).__init__()
        z_tilde = np.random.normal(size=(128, 2))
        self.z_tilde = torch.nn.Parameter(torch.from_numpy(z_tilde).float(), requires_grad=True)

    def forward(self, decoder):
        return decoder(self.z_tilde)


class SimpleClassifier(nn.Module):
    def __init__(self, in_channel=1, h_channels=[16, 16], n_classes=4):
        super(SimpleClassifier, self).__init__()

        self.in_channel = in_channel
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(in_channel, h_channels[0], 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(h_channels[0], h_channels[1], 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(h_channels[1], h_channels[1] // 2)
        self.fc2 = nn.Linear(h_channels[1] // 2, n_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.maxpool1(self.conv1(x))
        out = self.maxpool2(self.conv2(out))

        out = out.view(out.size(0), -1)

        out = self.relu(self.fc2(self.relu(self.fc1(out))))
        return out


class ClassifierTrainer(nn.Module):
    def __init__(self, cuda_num, train_loader, test_loader, h_channels=[16, 16], EPOCH=30, mode='ae'):
        super(ClassifierTrainer, self).__init__()
        self.EPOCH = EPOCH
        self.device = 'cuda:%d' % cuda_num
        self.classifier = SimpleClassifier(h_channels=h_channels)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.1, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.best_model = None
        self.best_loss = 999999

    def train(self, regularize=False, ratio=0., AT=False, epsilon=1.0, alpha=0.07, k=7, onoff=False, onadv_loader=None):
        self.classifier.to(self.device)

        adv = LinfPGDAttack(self.classifier, epsilon=epsilon, alpha=alpha, k=k)

        for epoch in range(self.EPOCH):
            train_loss = self.iter(epoch, self.train_loader, adv=adv, regularize=regularize, ratio=ratio, AT=AT,
                                   onoff=onoff, onadv_loader=onadv_loader)
            test_loss = self.iter(epoch, self.test_loader, train=False)

            print('epoch {0} summary: train_loss: {1}, test_loss: {2}'.format(epoch, train_loss, test_loss))

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.best_model = copy.deepcopy(self.classifier)

    def iter(self, epoch, loader, adv=None, train=True, regularize=False, ratio=0., AT=False, onoff=False,
             onadv_loader=None):
        if train:
            self.classifier.train()
        else:
            self.classifier.eval()
        total_loss = AverageMeter()

        if onoff:
            for i, ((X, y), (X2, y2)) in enumerate(zip(loader, onadv_loader)):
                X, y = X.to(self.device), y.long().to(self.device)
                X2, y2 = X2.to(self.device), y2.long().to(self.device)
                X1 = adv.perturb(X, y)

                out = self.classifier(X1)
                out2 = self.classifier(X2)

                out = torch.cat((out, out2))
                ys = torch.cat((y, y2))

                l = self.criterion(out, ys)
                total_loss.update(l.item(), ys.size(0))

                if train:
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()

                if regularize and ratio > 0.:
                    mask, lam = structured_prune(self.classifier, ratio)

                if i % 100 == 0:
                    print('epoch: {0}\tround: {1}\tloss: {2}'.format(epoch, i, l.item()))

        else:
            for i, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.long().to(self.device)
                if AT and train:
                    # print(AT)
                    X = adv.perturb(X, y)

                out = self.classifier(X)

                l = self.criterion(out, y)
                total_loss.update(l.item(), y.size(0))
                if train:
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()

                if regularize and ratio > 0.:
                    mask, lam = structured_prune(self.classifier, ratio)

                if i % 100 == 0:
                    print('epoch: {0}\tround: {1}\tloss: {2}'.format(epoch, i, l.item()))

        if regularize and ratio > 0.:
            mask, lam = structured_prune(self.classifier, ratio)

        return total_loss.avg


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, mode='ae'):
        super(Encoder, self).__init__()
        assert mode in ['ae', 'vae']
        self.mode = mode
        self.fc1 = nn.Linear(in_dim, 8)
        self.fc2 = nn.Linear(8, 16)
        self.relu = nn.LeakyReLU(0.2)
        self.meanfc = nn.Linear(16, out_dim)
        self.varfc = nn.Linear(16, out_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        if self.mode == 'vae':
            mean = self.meanfc(x)
            var = self.varfc(x)
            return mean, var
        else:
            z = self.meanfc(x)
            return z


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.in_fc = nn.Linear(in_dim, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, out_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.in_fc(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, sample_dim, device, mode='ae'):
        super(VAE, self).__init__()
        assert mode in ['ae', 'vae']
        self.mode = mode
        self.sample_dim = sample_dim
        self.device = device
        self.encoder = Encoder(latent_dim, sample_dim, mode)
        self.decoder = Decoder(sample_dim, latent_dim)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def reparameterize(self, mu, var):
        std = var.mul(0.5).exp_()
        epsilon = Variable(torch.randn(mu.size(0), self.sample_dim)).to(self.device)
        return mu + std * epsilon

    def forward(self, x):
        batch_size = x.size(0)
        if self.mode == 'vae':
            z_mean, z_var = self.encoder(x)
            z = self.reparameterize(z_mean, z_var)
            x_tilda = self.decoder(z)
            return z_mean, z_var, x_tilda
        else:
            z = self.encoder(x)
            x_tilda = self.decoder(z)
            return z, x_tilda

    def generate(self, x):
        batch_size = x.size(0)
        self.encoder.eval()
        if self.mode == 'vae':
            z_mean, z_var = self.encoder(x)
            z = self.reparameterize(z_mean, z_var)
        else:
            z = self.encoder(x)
        return z


class dataGenTrainer(nn.Module):
    def __init__(self, cuda_num, mode='ae', epochs=30, latent_dim=2, sample_dim=16, n_per_class=500, batch_size=128,
                 shuffle=True, lr=0.001, dist_info=None):
        super(dataGenTrainer, self).__init__()
        self.mode = mode
        self.EPOCH = epochs
        self.device = 'cuda:%d' % cuda_num
        self.vae = VAE(latent_dim, sample_dim, self.device, mode)
        self.X, self.y = make_classification_data(dist_info, n_per_class)
        self.data_loader = get_dataloader(self.X, self.y, batch_size, shuffle=shuffle)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=2e-4)
        self.l_list = []

    def recon_loss(self, x, x_tilde):
        return torch.mul(x - x_tilde, x - x_tilde)

    def loss(self, x, x_tilde, mus, log_variances):
        reconle = self.recon_loss(x, x_tilde)
        # kl divergence
        kl = -0.5 * torch.sum(-log_variances.exp() - torch.pow(mus, 2) + log_variances + 1, 1)
        return reconle.sum(1).mean() + kl.mean(), reconle.sum(1).mean(), kl.mean()

    def train(self):
        self.vae.to(self.device)
        for epoch in range(self.EPOCH):
            self.iter(epoch)

    def iter(self, epoch):
        self.vae.train()
        for i, (X, y) in enumerate(self.data_loader):
            X, y = X.to(self.device), y.to(self.device)
            if self.mode == 'vae':
                mu, var, x_tilda = self.vae(X)
                l, reconloss, klloss = self.loss(X, x_tilda, mu, var)
            else:
                z, x_tilda = self.vae(X)
                l = self.recon_loss(X, x_tilda).sum(1).mean()

            self.l_list.append(l.item())

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            if i % 50 == 0:
                if self.mode == 'vae':
                    print('epoch: {0}\tround: {1}\tloss: {2}[kl: {3}, recon: {4}]'.format(epoch, i, l.item(), klloss,
                                                                                          reconloss))
                else:
                    print('epoch: {0}\tround: {1}\tloss: {2}'.format(epoch, i, l.item()))

    def generate_data(self):
        self.vae.eval()

        X_generated_list = None
        y_generated_list = None

        for X, y in self.data_loader:
            X, y = X.to(self.device), y.to(self.device)
            gen_x = self.vae.generate(X)
            gen_x = gen_x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            X_generated_list = concatenate(X_generated_list, gen_x)
            y_generated_list = concatenate(y_generated_list, y)

        return X_generated_list, y_generated_list


def get_dataloader(X, y, batch_size: int = 128, drop_last=False, shuffle=True) -> DataLoader:
    X, y = torch.Tensor(X), torch.Tensor(y)
    loader = DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return loader


def make_classification_data(dist_info, n_per_class=500):
    X = None
    y = None
    for i in range(4):
        x = make_dummy_data(dist_info[i]['mean'], dist_info[i]['cov'], num_data=n_per_class)
        X = concatenate(X, x)
        y = concatenate(y, np.zeros((x.shape[0])) + i)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


# def make_classification_data(n_per_class=500):
#     mean = [0.1, -0.2]
#     cov = [[0.003, 0.005], [0.0025, -0.002]]
#     x1 = make_dummy_data(mean, cov, num_data=n_per_class)
#     mean = [-0., 0.1]
#     cov = [[0.02, 0.02], [0.002, -0.002]]
#     x2 = make_dummy_data(mean, cov, num_data=n_per_class)
#     mean = [0.3, 0.1]
#     cov = [[0.002, 0.004], [0.002, -0.002]]
#     x3 = make_dummy_data(mean, cov, num_data=n_per_class)
#     mean = [-0.2, 0.3]
#     cov = [[0.003, 0.0025], [0.004, -0.002]]
#     x4 = make_dummy_data(mean, cov, num_data=n_per_class)
#
#     X = np.concatenate((x1, x2, x3, x4))
#     y = np.concatenate(
#         (np.zeros((x1.shape[0])), np.ones((x1.shape[0])), np.ones((x1.shape[0])) * 2, np.ones((x1.shape[0])) * 3))
#
#     idx = np.arange(X.shape[0])
#     np.random.shuffle(idx)
#
#     return X[idx], y[idx]


def plot_jointdist(X, y, idxs=(0, 1), classes=4, kind='kde', numbers=1000):
    sns.set()

    if X.shape[0] > numbers:
        data = pd.DataFrame({'x1': X[:numbers, 0], 'x2': X[:numbers, 1], 'y': y[:numbers]})
    else:
        data = pd.DataFrame({'x1': X[:, idxs[0]], 'x2': X[:, idxs[1]], 'y': y[:]})
    sns.jointplot(data=data, x='x1', y='x2', hue='y', kind=kind, palette=sns.color_palette("Set2", classes))


def make_dummy_data(mean=[0, 0], cov=[[1, 2], [2, 3]], num_data=100):
    x, y = np.random.multivariate_normal(mean, cov, num_data).T
    data = np.concatenate(([x], [y]), axis=0).T
    return data


def plot_testdata(data):
    plt.plot(data[:, 0], data[:, 1], 'x')
    plt.axis('equal')
    plt.show()


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
            x = self.decoder(z + zeta)
            x = torch.clamp(x, 0, 1)

        return z, z_pert, x
