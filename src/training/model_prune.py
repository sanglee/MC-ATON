#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/08 3:05 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : model_prune.py
# @Software  : PyCharm
import torch
import torch.nn as nn

def structured_har(net, sparsity, mediate=16):
    idxs = {}
    lams = {}
    isFirst = True

    for i, conv in enumerate(net.conv_layers):
        if isinstance(conv, torch.nn.Conv1d):
            if isFirst:
                idx, lam = get_prun_idx(conv, sparsity)
                idxs['conv%d'%(i+1)] = idx
                lams['conv%d'%(i+1)] = lam
                conv_batch_prun(conv, idx)
            else:
                conv_batch_prun(conv, idx, structured=True, dim=1)
                idx, lam = get_prun_idx(conv, sparsity)
                idxs['conv%d'%(i+1)] = idx
                lams['conv%d'%(i+1)] = lam
                conv_batch_prun(conv, idx)

    pre_idx = idx.repeat_interleave(mediate)

    idxs['fc_pre'] = pre_idx
    fc_prun(net.fc_layers[0], pre_idx)
    idx, lam = get_prun_idx(net.fc_layers[0], sparsity)
    idxs['fc1'] = idx
    lams['fc1'] = lam
    fc_prun(net.fc_layers[0], idx, structured=False)
    fc_prun(net.fc_layers[2], idx)
    idx, lam = get_prun_idx(net.fc_layers[2], sparsity)
    idxs['fc2'] = idx
    lams['fc2'] = lam
    fc_prun(net.fc_layers[2], idx, structured=False)
    fc_prun(net.fc_layers[4], idx)

    return idxs, lams


def structured_prune(net, sparsity):
    isFirst = True
    idxs = {}
    lams = {}

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv' in name:
            if isFirst:
                isFirst = False
            else:
                conv_batch_prun(module, idx, structured=True, dim=1)
            idx, lam = get_prun_idx(module, sparsity)
            idxs[name] = idx
            lams[name] = lam
            conv_batch_prun(module, idx)

        if isinstance(module, nn.Linear) and name != 'fc2':
            fc_prun(module, idx)
            idx, lam = get_prun_idx(module, sparsity)
            idxs[name] = idx
            lams[name] = lam
            fc_prun(module, idx, structured=False)
        if isinstance(module, nn.Linear) and name == 'fc2':
            fc_prun(module, idx)

    return idxs, lams


def structured_lenet(net, sparsity, mediate=25):
    idxs = {}
    lams = {}
    isFirst = True
    for i, conv in enumerate(net.convs):
        if isinstance(conv, torch.nn.Conv2d):
            if isFirst:
                idx, lam = get_prun_idx(conv, sparsity)
                idxs['conv%d'%(i+1)] = idx
                lams['conv%d'%(i+1)] = lam
                conv_batch_prun(conv, idx)
            else:
                conv_batch_prun(conv, idx, structured=True, dim=1)
                idx, lam = get_prun_idx(conv, sparsity)
                idxs['conv%d'%(i+1)] = idx
                lams['conv%d'%(i+1)] = lam
                conv_batch_prun(conv, idx)

    pre_idx = idx.repeat_interleave(mediate)
    idxs['fc_pre'] = pre_idx
    fc_prun(net.fcs.fc1, pre_idx)
    idx, lam = get_prun_idx(net.fcs.fc1, sparsity)
    idxs['fc1'] = idx
    lams['fc1'] = lam
    fc_prun(net.fcs.fc1, idx, structured=False)
    fc_prun(net.fcs.fc2, idx)
    idx, lam = get_prun_idx(net.fcs.fc2, sparsity)
    idxs['fc2'] = idx
    lams['fc2'] = lam
    fc_prun(net.fcs.fc2, idx, structured=False)
    fc_prun(net.fcs.fc3, idx)

    return idxs, lams

def structured_resnet(net, sparsity):
    idxs = {}
    lams = {}

    idx, lam = get_prun_idx(net.conv1, sparsity)
    conv_batch_prun(net.conv1, idx)
    conv_batch_prun(net.bn1, idx, structured=False)

    idxs['pre'] = idx
    lams['pre'] = lam

    pre_idx = idx
    for li, layer in enumerate([net.layer1, net.layer2, net.layer3, net.layer4]):
        idxs['layer%d' % (li + 1)] = []
        lams['layer%d' % (li + 1)] = []
        for i, bb in enumerate(layer):
            if bb.downsample != None:
                pre_idx, pre_lam = basic_block_prune(bb, pre_idx, sparsity, isdown=True)
                idxs['layer%d' % (li + 1)].append(pre_idx)
                lams['layer%d' % (li + 1)].append(pre_lam)
            else:
                idx, lam = basic_block_prune(bb, pre_idx, sparsity)
                idxs['layer%d' % (li + 1)].append(idx)
                lams['layer%d' % (li + 1)].append(lam)
    fc_prun(net.fc, pre_idx)

    return idxs, lams


def compress_resnet(net, idxs):
    compress(net.conv1, idxs['pre'], dim=0)
    net.bn1 = compress(net.bn1, idxs['pre'], dim=0, bn=True)

    pre_idx = idxs['pre']
    for li, layer in enumerate([net.layer1, net.layer2, net.layer3, net.layer4]):
        for i, bb in enumerate(layer):
            block_idx = idxs['layer%d' % (li + 1)][i]
            if bb.downsample != None:
                basic_block_compress(bb, pre_idx, block_idx, isdown=True)
                pre_idx = block_idx
            else:
                basic_block_compress(bb, pre_idx, block_idx)
    compress(net.fc, pre_idx, dim=1)


def basic_block_compress(block, pre_idx, block_idx, isdown=False):
    if isdown:
        _ = compress(block.downsample[0], pre_idx, dim=1)
        compress(block.conv1, pre_idx, dim=1)

        compress(block.downsample[0], block_idx, dim=0)
        block.downsample[1] = compress(block.downsample[1], block_idx, dim=0, bn=True)
        compress(block.conv1, block_idx, dim=0)
        block.bn1 = compress(block.bn1, block_idx, dim=0, bn=True)

        compress(block.conv2, block_idx, dim=1)
        compress(block.conv2, block_idx, dim=0)
        block.bn2 = compress(block.bn2, block_idx, dim=0, bn=True)
    else:
        compress(block.conv1, pre_idx, dim=1)
        compress(block.conv1, block_idx, dim=0)
        block.bn1 = compress(block.bn1, block_idx, dim=0, bn=True)

        compress(block.conv2, block_idx, dim=1)
        compress(block.conv2, pre_idx, dim=0)
        block.bn2 = compress(block.bn2, pre_idx, dim=0, bn=True)


def compress(layer, idxs, dim=0, bn=False):
    assert dim in [1, 0]
    with torch.no_grad():
        if dim == 0:
            layer.weight = nn.Parameter(layer.weight.data[idxs == 1])
            if bn:
                param = nn.Parameter(layer.weight.data)
                bias = nn.Parameter(layer.bias.data[idxs == 1])
                layer = nn.BatchNorm2d(param.size(0), eps=layer.eps, momentum=layer.momentum, affine=layer.affine,
                                       track_running_stats=layer.track_running_stats)
                layer.weight = param
                layer.bias = bias
        elif dim == 1:
            layer.weight = nn.Parameter(layer.weight.data[:, idxs == 1])

        if layer.bias != None and dim == 0 and not bn:
            layer.bias = nn.Parameter(layer.bias.data[idxs == 1])
    return layer


def get_multiple_layer_prune(layers, sparsity):
    with torch.no_grad():
        for i, layer in enumerate(layers):
            if i == 0:
                p = layer.weight.data
                p = torch.linalg.norm(p.view(p.size(0), -1), dim=-1)
                p = p.unsqueeze(1)
            else:
                temp = layer.weight.data
                temp = torch.linalg.norm(temp.view(temp.size(0), -1), dim=-1)
                p = torch.cat((p, temp.unsqueeze(1)), dim=1)

        p = torch.linalg.norm(p, dim=1)

        if sparsity < 1.:
            lambda_ = p.sort(descending=True)[0][int(sparsity * (p.shape[0] - 1))]
            idxs = torch.ones_like(p)
            idxs[p.abs() < lambda_] = 0
        else:
            lambda_ = 0.
            idxs = torch.zeros_like(p)

    return idxs, lambda_


def basic_block_prune(block, pre_idx, sparsity, isdown=False):
    block_idx = {}
    block_lam = {}

    if isdown:
        conv_batch_prun(block.downsample[0], pre_idx, dim=1)
        conv_batch_prun(block.conv1, pre_idx, dim=1)

        idx, lam = get_multiple_layer_prune([block.conv1, block.downsample[0]], sparsity)
        block_idx = idx
        block_lam = lam

        conv_batch_prun(block.downsample[0], idx, dim=0)
        conv_batch_prun(block.downsample[1], idx, structured=False, dim=0)
        conv_batch_prun(block.conv1, idx, dim=0)
        conv_batch_prun(block.bn1, idx, structured=False, dim=0)

        conv_batch_prun(block.conv2, idx, dim=1)
        conv_batch_prun(block.conv2, idx, dim=0)
        conv_batch_prun(block.bn2, idx, structured=False, dim=0)

    else:
        conv_batch_prun(block.conv1, pre_idx, dim=1)
        idx, lam = get_prun_idx(block.conv1, sparsity)
        conv_batch_prun(block.conv1, idx, dim=0)
        conv_batch_prun(block.bn1, idx, structured=False, dim=0)
        block_idx = idx
        block_lam = lam

        conv_batch_prun(block.conv2, idx, dim=1)
        conv_batch_prun(block.conv2, pre_idx, dim=0)
        conv_batch_prun(block.bn2, pre_idx, structured=False, dim=0)

    return block_idx, block_lam


def get_prun_idx(layer, sparsity, structured=True, criterion='standard', rand_per=0.1):
    assert criterion in ['random', 'large', 'oursr', 'oursl', 'standard'], 'wrong criterion. (random, ours, standard)'

    with torch.no_grad():
        if structured:
            p = layer.weight.data
            p = p.view(p.size(0), -1)
            p = torch.linalg.norm(p, dim=1)
            if sparsity < 1.:
                lambda_ = p.sort()[0][int(sparsity * (p.shape[0]))]
                conv_idxs = torch.ones_like(p)
                conv_idxs[p.abs() < lambda_] = 0
            else:
                conv_idxs = torch.zeros_like(p)
                lambda_ = 0.
            # if criterion == 'random':
            #     perm = torch.randperm(p.size(0))
            #     idx = perm[:int(p.size(0) * 0.5)]
            #     conv_idxs = torch.ones_like(p)
            #     conv_idxs[idx] = 0
            #     lambda_ = 0.
            # elif criterion == 'oursr':
            #     lambda_ = p.sort()[0][int(sparsity * rand_per * p.size(0))]
            #     idx1 = p.sort()[1][:int(p.size(0) * sparsity * (1 - rand_per))]
            #
            #     perm = torch.randperm(p.size(0) - int(p.size(0) * sparsity * (1 - rand_per)))
            #     perm = perm[:int(p.size(0) * sparsity * rand_per)]
            #     idx2 = p.sort()[1][int(p.size(0) * sparsity * (1 - rand_per)):][perm]
            #
            #     conv_idxs = torch.ones_like(p)
            #     conv_idxs[torch.cat((idx1, idx2))] = 0
            # elif criterion == 'standard':
            #     # print(int(sparsity * (p.shape[0] - 1)), p.shape[0]-1, p.sort()[0])
            #     if sparsity < 1.:
            #         lambda_ = p.sort()[0][int(sparsity * (p.shape[0]))]
            #         conv_idxs = torch.ones_like(p)
            #         conv_idxs[p.abs() < lambda_] = 0
            #     else:
            #         conv_idxs = torch.zeros_like(p)
            #         lambda_ = 0.
            #
            # elif criterion == 'large':
            #     lambda_ = p.sort(descending=True)[0][int(sparsity * p.shape[0])]
            #     conv_idxs = torch.ones_like(p)
            #     conv_idxs[p.abs() > lambda_] = 0
            # elif criterion == 'oursl':
            #     lambda_ = p.sort()[0][int(sparsity * rand_per * p.size(0))]
            #     idx1 = p.sort()[1][:int(p.size(0) * sparsity * (1 - rand_per))]
            #     idx2 = p.sort()[1][p.size(0) - int(p.size(0) * sparsity * rand_per):-1]
            #     conv_idxs = torch.ones_like(p)
            #     conv_idxs[torch.cat((idx1, idx2))] = 0
        else:
            p = layer.weight.abs()
            lambda_ = p.view(-1).sort()[0][int(sparsity * (p.view(-1).shape[0]-1))]
            conv_idxs = torch.ones_like(p)
            conv_idxs[p <= lambda_] = 0

    return conv_idxs, lambda_


def conv_batch_prun(layer, idxs, structured=True, dim=0):
    with torch.no_grad():
        if structured:
            if dim == 0:
                layer.weight[idxs == 0, :, :] = 0
            if dim == 1:
                layer.weight[:, idxs == 0, :] = 0
        else:
            layer.weight[idxs == 0] = 0

        if layer.bias != None and dim == 0:
            layer.bias[idxs == 0] = 0


def fc_prun(layer, idxs, structured=True):
    with torch.no_grad():
        if structured:
            layer.weight[:, idxs == 0] = 0
        else:
            layer.weight[idxs == 0] = 0
