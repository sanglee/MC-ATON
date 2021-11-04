#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/09 1:30 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : off_manifold_attack.py
# @Software  : PyCharm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm_notebook as tqdm

import attacker
from data import cifar10
from models import model_loader
from utils import save_data, load_data, accuracy, plot_grid


def get_success(net, loader, device):
    X_success = []
    y_success = []
    X_unsuccess = []
    y_unsuccess = []

    total = 0

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        total += X.size(0)

        out = net(X)

        out = F.softmax(out)
        mask = (y == torch.argmax(out, dim=1)).nonzero(as_tuple=True)[0]
        mask2 = (y != torch.argmax(out, dim=1)).nonzero(as_tuple=True)[0]

        if i == 0:
            X_success = X[mask]
            y_success = y[mask]
            X_unsuccess = X[mask2]
            y_unsuccess = y[mask2]

        else:
            X_success = torch.cat((X_success, X[mask]), dim=0)
            y_success = torch.cat((y_success, y[mask]), dim=0)
            X_unsuccess = torch.cat((X_unsuccess, X[mask2]), dim=0)
            y_unsuccess = torch.cat((y_unsuccess, y[mask2]), dim=0)

    success_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_success, y_success),
        batch_size=100, shuffle=False)
    unsuccess_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_unsuccess, y_unsuccess),
        batch_size=100, shuffle=False)

    return success_loader, unsuccess_loader, X_success.size(0) / total * 100


def off_manifold_attack(net, loader, sparsity_ratio, base_path, attack_type, attack_args, device, save=False, plot=False):

    origin_net = model_loader(model_name='resnet34', pretrain=True)
    compressed_net = model_loader(model_name='resnet34', pretrain=True)

    # untargeted
    if attack_type == 'FGSM':
        attack = getattr(attacker, attack_type)(net, config=attack_args)
    else:
        attack = getattr(attacker, attack_type)(net, **attack_args)

    x_adv_list = []
    pert_list = []
    y_list = []
    x_list = []
    returnplot = []
    for i, (X, y) in enumerate(tqdm(loader)):
        X, y = X.to(device), y.to(device)

        if attack_type == 'FGSM':
            adv, pert = attack(X, y)
        else:
            adv, pert = attack.perturb(X, y)

        if i == 0:
            x_adv_list, pert_list, y_list, x_list = adv.cpu(), pert.cpu(), y.cpu(), X.cpu()
        else:
            x_adv_list = torch.cat((x_adv_list, adv.cpu()))
            pert_list = torch.cat((pert_list, pert.cpu()))
            y_list = torch.cat((y_list, y.cpu()))
            x_list = torch.cat((x_list, X.cpu()))

    if save:
        file = '{}/{}_{}/'.format(base_path, 'cifar10', 'resnet34')
        save_data((x_adv_list, pert_list, x_list, y_list), file,
                  filename='adv_data_sparsity%d.pkl' % (sparsity_ratio * 100))

    if plot:
        returnplot = torch.cat((x_adv_list[:16], pert_list[:16], x_list[:16])).cpu()
        plot_grid(returnplot)

    return x_adv_list, pert_list, x_list, y_list, returnplot


def attack_score(x_adv_list, y_list, net, sparsity_ratio, device, base_path, save_result=False):
    # file = '{}/{}_{}/{}'.format(BASE_PATH, 'cifar10', 'resnet34', 'adv_data_%d.pkl' % (ratio * 100))
    # x_adv_list, pert_list, x_list, y_list = load_data(file)

    adv_loader = DataLoader(
        torch.utils.data.TensorDataset(x_adv_list, y_list),
        batch_size=100, shuffle=False)

    # net = model_loader(model_name='resnet34', pretrain=False)
    # file = '{}/{}_{}/{}'.format(BASE_PATH, 'cifar10', 'resnet34', 'benign_%d.pkl' % (ratio * 100))
    # comp = load_data(file)
    # net.load_state_dict(comp['state_dict'])

    # benign_pred = []
    pert_pred = []
    # benign_label = []
    pert_label = []

    net.eval()

    for idx, (X_adv, y) in enumerate(adv_loader):

        pert_out = net(X_adv.to(device))

        if idx == 0:
            # benign_pred = F.softmax(benign_out)
            pert_pred = F.softmax(pert_out)
            # _, tmp_benign_y = benign_pred.data.max(dim=1)
            _, tmp_pert_y = pert_pred.data.max(dim=1)

            # benign_pred = benign_pred.detach().cpu()
            pert_pred = pert_pred.detach().cpu()
            # benign_label = tmp_benign_y.detach().cpu()
            pert_label = tmp_pert_y.detach().cpu()

        else:
            # tmp_benign_pred = F.softmax(benign_out)
            tmp_pert_pred = F.softmax(pert_out)
            # _, tmp_benign_y = tmp_benign_pred.data.max(dim=1)
            _, tmp_pert_y = tmp_pert_pred.data.max(dim=1)

            # benign_pred = torch.cat((benign_pred, tmp_benign_pred.detach().cpu()))
            pert_pred = torch.cat((pert_pred, tmp_pert_pred.detach().cpu()))
            # benign_label = torch.cat((benign_label, tmp_benign_y.detach().cpu()))
            pert_label = torch.cat((pert_label, tmp_pert_y.detach().cpu()))

    # benign_score = accuracy(y_list.numpy(), benign_label.numpy(), benign_pred.numpy())
    adv_score = accuracy(y_list.numpy(), pert_label.numpy(), pert_pred.numpy())

    print('Compression Ratio: %d%%' % (sparsity_ratio * 100))
    # print('benign_score: \t', benign_score)
    print('attack_score: \t', adv_score)

    if save_result:
        file = '{}/{}_{}/'.format(base_path, 'cifar10', 'resnet34')
        save_data((pert_pred, pert_label, adv_score), file,
                  filename='offmanifold_result_sparsity%d.pkl' % sparsity_ratio * 100)

    # return benign_pred, pert_pred, benign_label, pert_label, benign_score, adv_score
    return pert_pred, pert_label, adv_score


# if __name__ == '__main__':
#     base_path = '/workspace/paper_works/work_results/finally'
#     sparsity_ratio = .1
#     device = 'cuda:%d' % 5
#
#     pgd_args = {
#         "eps": 8.0/255.,
#         "attack_steps": 30,
#         "attack_lr": 0.02,
#         "random_init": True,
#     }
#
#     trainset, testset, _ = cifar10()
#
#     net = model_loader(model_name='resnet34', pretrain=False)
#     file = '{}/{}_{}/{}'.format(base_path, 'cifar10', 'resnet34', 'benign_%d.pkl' % (sparsity_ratio * 100))
#     comp = load_data(file)
#     net.load_state_dict(comp['state_dict'])
#     net.to(device)
#
#     success_set, ratio = get_success(net, testset, device)
#
#     x_adv_list, pert_list, x_list, y_list = off_manifold_attack(net, success_set, sparsity_ratio, base_path, pgd_args,
#                                                                 save=False, plot=False)
#
#     benign_pred, pert_pred, benign_label, pert_label, benign_score, adv_score = attack_score(x_adv_list, pert_list,
#                                                                                              x_list, y_list, net,
#                                                                                              sparsity_ratio, device,
#                                                                                              base_path,
#                                                                                              save_result=False)


