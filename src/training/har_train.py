#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/10/28 17:12
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : har_train.py
# @Software  : PyCharm

import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.auto import tqdm

from attacker import LinfPGDAttack
from data import uci_har
from models import HARClassifier
from training import structured_har
from utils import AverageMeter


def train_iter(model, optimizer, criterion, data_loader, device, mode=0, comp_ratio=0.):
    model.train()
    iteration_loss = AverageMeter()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        output = model(X)

        loss = criterion(output, y.long())

        iteration_loss.update(loss.item(), X.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if comp_ratio > 0:
            idxs, lams = structured_har(model, comp_ratio)
    if comp_ratio > 0:
        idxs, lams = structured_har(model, comp_ratio)
    return iteration_loss.avg


def test_iter(model, data_loader, device, check_adv=False, epsilon=0.3, alpha=0.0073, k=7):
    model.eval()
    normal_acc = AverageMeter()

    if check_adv:
        adv = LinfPGDAttack(model, epsilon=epsilon, alpha=alpha, k=k)
        off_acc = AverageMeter()

    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        output = model(X)

        out = F.softmax(output, dim=1)
        _, predicted = out.max(1)
        idx = predicted.eq(y)

        acc = idx.sum().item() / X.size(0)
        normal_acc.update(acc)

        if check_adv:
            adv_x = adv.perturb(X, y.long())

            out = model(adv_x)
            out = F.softmax(out, dim=1)
            _, predicted = out.max(1)
            idx = predicted.eq(y)

            acc = idx.sum().item() / X.size(0)
            off_acc.update(acc)

    if check_adv:
        return normal_acc.avg, off_acc.avg
    else:
        return normal_acc.avg


def har_train(cuda_num=4, EPOCH=100, save_interval=5, resume=True, comp_ratio=0., model_dir='./simulation/HAR_UCI/'):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    train_loader, test_loader = uci_har('/workspace/Dataset/TSData/uci_data/np/')
    device = 'cuda:%d' % cuda_num

    model = HARClassifier()
    model.to(device)
    if resume:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'comp0_0-model-epoch{}.pt'.format(99))))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.91)
    scheduler = MultiStepLR(optimizer, milestones=[55, 75, 90], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(EPOCH)):
        train_loss = train_iter(model, optimizer, criterion, train_loader, device, comp_ratio)
        val_acc = test_iter(model, test_loader, device)
        scheduler.step()

        print('Epoch {}\tTrain loss: {:.4f}\tValidation accuracy: {:.4f}'.format(epoch + 1, train_loss, val_acc))

        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir,
                                    'comp{}-model-epoch{}.pt'.format(str(comp_ratio * 100).replace('.', '_'), epoch)))
