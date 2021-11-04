#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/21 5:17 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : vae_train.py
# @Software  : PyCharm

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
from torchvision.utils import make_grid

from attacker import LinfPGDAttack
from data import mnist
from models import model_loader, VAE_2d, L0LeNet, LeNet_3C
from training import structured_resnet, structured_lenet, OnManifoldPerturbation
from utils import AverageMeter


def show_and_save(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    fig = plt.figure(dpi=200)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')
    plt.imshow(npimg)
    plt.imsave(f, npimg)


def iteration(epoch, epochs, gen, discrim, loader, optim_e, optim_d, optim_Dis, criterion, device, gamma):
    gen.train()
    discrim.train()

    recon_loss_list = AverageMeter()
    prior_loss_list = AverageMeter()
    dis_real_list = AverageMeter()
    dis_fake_list = AverageMeter()
    dis_prior_list = AverageMeter()
    gan_loss_list = AverageMeter()

    for i, (data, _) in enumerate(loader):
        bs = data.size()[0]

        ones_label = Variable(torch.ones(bs, 1)).to(device)
        zeros_label = Variable(torch.zeros(bs, 1)).to(device)
        zeros_label1 = Variable(torch.zeros(64, 1)).to(device)
        datav = Variable(data).to(device)
        mean, logvar, rec_enc = gen(datav)
        z_p = Variable(torch.randn(64, 128)).to(device)
        x_p_tilda = gen.decoder(z_p)

        output = discrim(datav)[0]
        errD_real = criterion(output, ones_label)
        dis_real_list.update(errD_real.item(), data.size(0))
        output = discrim(rec_enc)[0]
        errD_rec_enc = criterion(output, zeros_label)
        dis_fake_list.update(errD_rec_enc.item(), data.size(0))
        output = discrim(x_p_tilda)[0]
        errD_rec_noise = criterion(output, zeros_label1)
        dis_prior_list.update(errD_rec_noise.item(), data.size(0))
        gan_loss = errD_real + errD_rec_enc + errD_rec_noise
        gan_loss_list.update(gan_loss.item(), data.size(0))
        optim_Dis.zero_grad()
        gan_loss.backward(retain_graph=True)
        optim_Dis.step()

        output = discrim(datav)[0]
        errD_real = criterion(output, ones_label)
        output = discrim(rec_enc)[0]
        errD_rec_enc = criterion(output, zeros_label)
        output = discrim(x_p_tilda)[0]
        errD_rec_noise = criterion(output, zeros_label1)
        gan_loss = errD_real + errD_rec_enc + errD_rec_noise

        x_l_tilda = gen.discriminator(rec_enc)[1]
        x_l = discrim(datav)[1]
        rec_loss = ((x_l_tilda - x_l) ** 2).mean()
        err_dec = gamma * rec_loss - gan_loss
        recon_loss_list.update(rec_loss.item(), data.size(0))
        optim_d.zero_grad()
        err_dec.backward(retain_graph=True)
        optim_d.step()

        mean, logvar, rec_enc = gen(datav)
        x_l_tilda = discrim(rec_enc)[1]
        x_l = discrim(datav)[1]
        rec_loss = ((x_l_tilda - x_l) ** 2).mean()
        prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
        prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
        prior_loss_list.update(prior_loss.item(), data.size(0))
        err_enc = prior_loss + 5 * rec_loss

        optim_e.zero_grad()
        err_enc.backward(retain_graph=True)
        optim_e.step()

        if i % 50 == 0:
            print(
                '[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f'
                % (epoch, epochs, i, len(loader),
                   gan_loss.item(), prior_loss.item(), rec_loss.item(), errD_real.item(), errD_rec_enc.item(),
                   errD_rec_noise.item()))


def train_vae(discrim, gan, device):
    data_loader, _, _ = mnist(128)
    real_batch = next(iter(data_loader))

    z_fixed = Variable(torch.randn((128, 128))).to(device)
    x_fixed = Variable(real_batch[0]).to(device)

    epochs = 25
    lr = 3e-4
    alpha = 0.1
    gamma = 15

    criterion = nn.BCELoss().to(device)
    optim_E = torch.optim.RMSprop(gan.encoder.parameters(), lr=lr)
    optim_D = torch.optim.RMSprop(gan.decoder.parameters(), lr=lr)
    optim_Dis = torch.optim.RMSprop(discrim.parameters(), lr=lr * alpha)

    for epoch in range(epochs):
        iteration(epoch, epochs, gan, discrim, data_loader, optim_E, optim_D, optim_Dis, criterion, device, gamma)

        b = gan(x_fixed)[2]
        b = b.detach()
        c = gan.decoder(z_fixed)
        c = c.detach()

        show_and_save('./simulation/results/MNISTrec_noise_epoch_%d.png' % epoch, make_grid((c * 0.5 + 0.5).cpu(), 8))
        show_and_save('./simulation/results/MNISTrec_epoch_%d.png' % epoch, make_grid((b * 0.5 + 0.5).cpu(), 8))


class VAETrainer(object):
    def __init__(self, data_loader, in_dim=3, cuda_num=5):
        self.device = 'cuda:%d' % cuda_num

        self.loader = data_loader

        self.epochs = 25
        self.lr = 3e-4
        self.alpha = 0.1
        self.gamma = 15

        self.gan = VAE_2d(self.device, in_dim=in_dim).to(self.device)
        # self.discrim = Discriminator().to(self.device)

        self.criterion = nn.BCELoss().to(self.device)
        self.optim_E = torch.optim.Adam(self.gan.parameters(), lr=self.lr)
        # self.optim_D = torch.optim.Adam(self.gan.decoder.parameters(), lr=self.lr)
        # self.optim_Dis = torch.optim.Adam(self.discrim.parameters(), lr=self.lr * self.alpha)

    def iterations(self, epoch):
        self.gan.train()
        # self.discrim.train()

        recon_loss_list = AverageMeter()
        prior_loss_list = AverageMeter()
        dis_real_list = AverageMeter()
        dis_fake_list = AverageMeter()
        dis_prior_list = AverageMeter()
        gan_loss_list = AverageMeter()

        for i, (data, _) in enumerate(self.loader):
            bs = data.size()[0]

            # ones_label = Variable(torch.ones(bs, 1)).to(self.device)
            # zeros_label = Variable(torch.zeros(bs, 1)).to(self.device)
            # zeros_label1 = Variable(torch.zeros(64, 1)).to(self.device)
            datav = Variable(data).to(self.device)
            # mean, logvar, rec_enc = self.gan(datav)
            # z_p = Variable(torch.randn(64, 128)).to(self.device)
            # x_p_tilda = self.gan.decoder(z_p)
            #
            # output = self.discrim(datav)[0]
            # errD_real = self.criterion(output, ones_label)
            # dis_real_list.update(errD_real.item(), data.size(0))
            # output = self.discrim(rec_enc)[0]
            # errD_rec_enc = self.criterion(output, zeros_label)
            # dis_fake_list.update(errD_rec_enc.item(), data.size(0))
            # output = self.discrim(x_p_tilda)[0]
            # errD_rec_noise = self.criterion(output, zeros_label1)
            # dis_prior_list.update(errD_rec_noise.item(), data.size(0))
            # gan_loss = errD_real + errD_rec_enc + errD_rec_noise
            # gan_loss_list.update(gan_loss.item(), data.size(0))
            # self.optim_Dis.zero_grad()
            # gan_loss.backward(retain_graph=True)
            # self.optim_Dis.step()
            #
            # output = self.discrim(datav)[0]
            # errD_real = self.criterion(output, ones_label)
            # output = self.discrim(rec_enc)[0]
            # errD_rec_enc = self.criterion(output, zeros_label)
            # output = self.discrim(x_p_tilda)[0]
            # errD_rec_noise = self.criterion(output, zeros_label1)
            # gan_loss = errD_real + errD_rec_enc + errD_rec_noise
            #
            # x_l_tilda = self.gan.discriminator(rec_enc)[1]
            # x_l = self.discrim(datav)[1]
            # rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            # err_dec = self.gamma * rec_loss - gan_loss
            # recon_loss_list.update(rec_loss.item(), data.size(0))
            # self.optim_D.zero_grad()
            # err_dec.backward(retain_graph=True)
            # self.optim_D.step()

            mean, logvar, rec_enc = self.gan(datav)
            # x_l_tilda = self.discrim(rec_enc)[1]
            # x_l = self.discrim(datav)[1]
            rec_loss = ((rec_enc - datav) ** 2).mean()
            prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
            prior_loss_list.update(prior_loss.item(), data.size(0))
            err_enc = prior_loss + 5 * rec_loss

            self.optim_E.zero_grad()
            err_enc.backward()
            self.optim_E.step()

            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_prior: %.4f\tRec_loss: %.4f'
                    % (epoch, self.epochs, i, len(self.loader),
                       prior_loss.item(), rec_loss.item()))

    #                 , gan_loss.item(), errD_real.item(), errD_rec_enc.item(),
    #                        errD_rec_noise.item())
    # \tLoss_gan: % .4
    # f\tdis_real_loss: % 0.4
    # f\tdis_fake_loss: % .4
    # f\tdis_prior_loss: % .4
    # f

    def train(self):
        for epoch in range(self.epochs):
            self.iterations(epoch)

            real_batch = next(iter(self.loader))

            z_fixed = Variable(torch.randn((128, 128))).to(self.device)
            x_fixed = Variable(real_batch[0]).to(self.device)

            b = self.gan(x_fixed)[2]
            b = b.detach()
            c = self.gan.decoder(z_fixed)
            c = c.detach()

            show_and_save('./simulation/results/MNISTrec_noise_epoch_%d.png' % epoch,
                          make_grid((c * 0.5 + 0.5).cpu(), 8))
            show_and_save('./simulation/results/MNISTrec_epoch_%d.png' % epoch, make_grid((b * 0.5 + 0.5).cpu(), 8))


class ClassifierTrainer(nn.Module):
    def __init__(self, model_type, cuda_num, train_loader, test_loader, EPOCH=30, lr=0.001, weight_decay=5e-6, gamma = 0.95, schedule_lr=False):
        super(ClassifierTrainer, self).__init__()
        self.EPOCH = EPOCH
        self.device = 'cuda:%d' % cuda_num
        self.model_type = model_type
        if model_type == 'lenet':
            self.classifier = L0LeNet(10, device=cuda_num)
        elif model_type == 'lenet_3c':
            self.classifier = LeNet_3C(10, device=cuda_num)
        else:
            self.classifier = model_loader('resnet18', 1, 10)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)

        self.criterion = nn.CrossEntropyLoss()
        self.best_model = None
        self.best_loss = 999999
        self.schedule_lr = schedule_lr


    def train(self, regularize=False, ratio=0., AT=False, onAT=False, epsilon=1.0, alpha=0.07, k=7, onoff=False,
              onmanifold_loader=None, gan=None):
        self.classifier.to(self.device)
        # if AT:
        adv = LinfPGDAttack(self.classifier, epsilon=epsilon, alpha=alpha, k=k)
        # if onAT:
        #     adv = OnManifoldPerturbation(self.classifier, gan, self.device, epsilon=epsilon, alpha=alpha, k=k)
        for epoch in range(self.EPOCH):
            train_loss = self.iter(epoch, self.train_loader, onoff=onoff, adv=adv, onmanifold_loader=onmanifold_loader,
                                   regularize=regularize, ratio=ratio, AT=AT, onAT=onAT)
            test_loss = self.iter(epoch, self.test_loader, adv=adv, train=False)
            if self.schedule_lr:
                self.scheduler.step()

            print('epoch {0} summary: train_loss: {1}, test_loss: {2}'.format(epoch, train_loss, test_loss))

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.best_model = copy.deepcopy(self.classifier)

    def iter(self, epoch, loader, onoff=False, adv=None, train=True, regularize=False, ratio=0., AT=False, onAT=False,
             onmanifold_loader=None):
        if train:
            self.classifier.train()
        else:
            self.classifier.eval()
        total_loss = AverageMeter()
        total_list = None

        if onoff:
            for i, ((X, y), (X2, y2)) in enumerate(zip(loader, onmanifold_loader)):
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
                    if self.model_type == 'lenet':
                        idxs, lams = structured_lenet(self.classifier, ratio)
                    elif self.model_type == 'lenet_3c':
                        idxs, lams = structured_lenet(self.classifier, ratio, 4)
                    else:
                        idxs, lams = structured_resnet(self.classifier, ratio)

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
                    if self.model_type == 'lenet':
                        idxs, lams = structured_lenet(self.classifier, ratio)
                    elif self.model_type == 'lenet_3c':
                        idxs, lams = structured_lenet(self.classifier, ratio, 4)
                    else:
                        idxs, lams = structured_resnet(self.classifier, ratio)

        if regularize and ratio > 0.:
            if self.model_type == 'lenet':
                idxs, lams = structured_lenet(self.classifier, ratio)
            elif self.model_type == 'lenet_3c':
                idxs, lams = structured_lenet(self.classifier, ratio, 4)
            else:
                idxs, lams = structured_resnet(self.classifier, ratio)

        return total_loss.avg

    # def train(self, regularize=False, ratio=0., AT=False, onAT=False, epsilon=1.0, alpha=0.07, k=7, onoff=False,
    #           onmanifold_loader=None, gan=None, l_clf=None):
    #     self.classifier.to(self.device)
    #     if AT or onoff:
    #         adv = LinfPGDAttack(self.classifier, epsilon=epsilon, alpha=alpha, k=k)
    #     if onAT or onoff:
    #         onadv = OnManifoldPerturbation(self.classifier, gan, self.device, epsilon=epsilon, alpha=alpha, k=k)
    #     for epoch in range(self.EPOCH):
    #         train_loss = self.iter(epoch, self.train_loader, onoff=onoff, adv=adv, onadv=onadv, onmanifold_loader=onmanifold_loader,
    #                                regularize=regularize, ratio=ratio, AT=AT, onAT=onAT, gan=gan, l_clf=l_clf)
    #         test_loss = self.iter(epoch, self.test_loader, adv=adv, train=False)
    #
    #         print('epoch {0} summary: train_loss: {1}, test_loss: {2}'.format(epoch, train_loss, test_loss))
    #
    #         if test_loss < self.best_loss:
    #             self.best_loss = test_loss
    #             self.best_model = copy.deepcopy(self.classifier)
    #
    # def iter(self, epoch, loader, onoff=False, adv=None, onadv=None, train=True, regularize=False, ratio=0., AT=False, onAT=False,
    #          onmanifold_loader=None, gan=None, l_clf=None):
    #     if train:
    #         self.classifier.train()
    #     else:
    #         self.classifier.eval()
    #     total_loss = AverageMeter()
    #     total_list = None
    #
    #     if onoff:
    #         for i, (X, y) in enumerate(loader):
    #             X, y = X.to(self.device), y.long().to(self.device)
    #                 # print(AT)
    #             X1 = adv.perturb(X, y)
    #             z, z_pert, X2 = onadv.perturb(X, y)
    #
    #             out, _ = gan.encoder(X2.to(self.device))
    #             out = l_clf(out.unsqueeze(1))
    #             out = F.softmax(out, dim=1)
    #             _, pred = out.max(1)
    #
    #             idx = pred.eq(y)
    #
    #             X2 = X[idx]
    #             y2 = y[idx]
    #
    #             out = self.classifier(X1)
    #             out2 = self.classifier(X2)
    #
    #             out = torch.cat((out, out2))
    #             ys = torch.cat((y, y2))
    #
    #             l = self.criterion(out, ys)
    #
    #             total_loss.update(l.item(), ys.size(0))
    #
    #             if train:
    #                 self.optimizer.zero_grad()
    #                 l.backward()
    #                 self.optimizer.step()
    #
    #             if regularize and ratio > 0.:
    #                 if self.model_type == 'lenet':
    #                     idxs, lams = structured_lenet(self.classifier, ratio)
    #                 else:
    #                     idxs, lams = structured_resnet(self.classifier, ratio)
    #
    #     else:
    #         for i, (X, y) in enumerate(loader):
    #             X, y = X.to(self.device), y.long().to(self.device)
    #             if AT and train:
    #                 # print(AT)
    #                 X = adv.perturb(X, y)
    #             if onAT and train:
    #                 z, z_pert, X = adv.perturb(X, y)
    #                 out, _ = gan.encoder(X.to(self.device))
    #                 out = l_clf(out.unsqueeze(1))
    #                 out = F.softmax(out, dim=1)
    #                 _, pred = out.max(1)
    #
    #                 idx = pred.eq(y)
    #
    #                 X = X[idx]
    #                 y = y[idx]
    #
    #             out = self.classifier(X)
    #
    #             l = self.criterion(out, y)
    #             total_loss.update(l.item(), y.size(0))
    #             if train:
    #                 self.optimizer.zero_grad()
    #                 l.backward()
    #                 self.optimizer.step()
    #
    #             if regularize and ratio > 0.:
    #                 if self.model_type == 'lenet':
    #                     idxs, lams = structured_lenet(self.classifier, ratio)
    #                 else:
    #                     idxs, lams = structured_resnet(self.classifier, ratio)
    #
    #             # if i % 50 == 0:
    #             #     print('epoch: {0}\tround: {1}\tloss: {2}'.format(epoch, i, l.item()))
    #
    #     if regularize and ratio > 0.:
    #         if self.model_type == 'lenet':
    #             idxs, lams = structured_lenet(self.classifier, ratio)
    #         else:
    #             idxs, lams = structured_resnet(self.classifier, ratio)
    #
    #     return total_loss.avg