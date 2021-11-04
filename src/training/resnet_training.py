#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/08 4:17 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : resnet_training.py
# @Software  : PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
# from tqdm.notebook import tqdm_notebook as tqdm
from tqdm import tqdm

import data
from attacker import LinfPGDAttack
from models import model_loader
from training import structured_resnet
from utils import AverageMeter, reshape_resulting_array, get_weights_copy, load_checkpoint
from utils import load_data, accuracy, save_data, save_checkpoint


def robust_train_iter(net, loader, adversary, criterion, optimizer, epoch, device, reg_ratio=0., print_freq=100):
    losses = AverageMeter()
    batch_time = AverageMeter()
    net.train()
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        adv = adversary.perturb(X, y)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, y)

        losses.update(loss.data, X.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if reg_ratio > 0.:
            idxs, lams = structured_resnet(net, reg_ratio)

        if not i == 0 and i % print_freq == 0 or len(loader) - 1 == i:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), loss=losses))

    if reg_ratio > 0.:
        idxs, lams = structured_resnet(net, reg_ratio)

    state_dicts = get_weights_copy(net, device, weights_path='tmp/weights_temp.pt')

    state = {
        'epoch': epoch + 1,
        'state_dict': state_dicts,
        'loss': losses.avg.detach().cpu().numpy().tolist(),
    }

    if reg_ratio > 0.:
        state['reg_idxs'] = idxs
        state['reg_lams'] = lams

    return state


def robust_valid_iter(net, loader, adversary, criterion, epoch, device, reg_ratio=0., print_freq=30):
    benign_losses = AverageMeter()
    adv_losses = AverageMeter()
    benign_correct = 0
    adv_correct = 0
    total = 0
    batch_time = AverageMeter()
    net.eval()
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        total += y.size(0)

        out = net(X)
        loss = criterion(out, y)
        benign_losses.update(loss.data, X.size(0))

        _, predicted = out.max(1)
        benign_correct += predicted.eq(y).sum().item()

        benign_acc = str(predicted.eq(y).sum().item() / y.size(0))

        adv = adversary.perturb(X, y)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, y)

        adv_losses.update(loss.data, X.size(0))
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(y).sum().item()

        adv_acc = str(predicted.eq(y).sum().item() / y.size(0))

        if not i == 0 and i % print_freq == 0 or len(loader) - 1 == i:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'benign accuracy: [{3}]\t'
                  'adversarial accuracy: [{4}]\t'
                  'benign loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'adversarial loss: {adv.val:.4f} ({adv.avg:.4f})\t'.format(
                epoch, i, len(loader), benign_acc, adv_acc, loss=benign_losses, adv=adv_losses))

    state_dicts = get_weights_copy(net, device, weights_path='tmp/weights_temp.pt')

    state = {
        'epoch': epoch + 1,
        'state_dict': state_dicts,
        'benign_loss': benign_losses.avg.detach().cpu().numpy().tolist(),
        'adversarial_loss': adv_losses.avg.detach().cpu().numpy().tolist(),
        'benign_acc': 100. * benign_correct / total,
        'adversarial_acc': 100. * adv_correct / total,
    }

    if reg_ratio > 0.:
        state['reg_idxs'] = idxs
        state['reg_lams'] = lams

    return state


def train_iter(net, loader, criterion, optimizer, epoch, device, reg_ratio=0., print_freq=100):
    losses = AverageMeter()
    batch_time = AverageMeter()
    net.train()
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        pred = net(X)
        loss = criterion(pred, y)

        losses.update(loss.data, X.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if reg_ratio > 0.:
            idxs, lams = structured_resnet(net, reg_ratio)

        if not i == 0 and i % print_freq == 0 or len(loader) - 1 == i:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), loss=losses))

    if reg_ratio > 0.:
        idxs, lams = structured_resnet(net, reg_ratio)

    state_dicts = get_weights_copy(net, device, weights_path='tmp/weights_temp.pt')

    state = {
        'epoch': epoch + 1,
        'state_dict': state_dicts,
        'loss': losses.avg.detach().cpu().numpy().tolist(),
    }

    if reg_ratio > 0.:
        state['reg_idxs'] = idxs
        state['reg_lams'] = lams

    return state


def valid_iter(net, loader, criterion, epoch, device, return_result=True, print_freq=30):
    losses = AverageMeter()
    net.eval()

    total_softmax = []
    total_labels = []
    total_preds = []

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        pred = net(X)
        loss = criterion(pred, y)

        losses.update(loss.data, X.size(0))

        if return_result:
            pred = F.softmax(pred)
            _, p_data = pred.data.max(dim=1)
            p_data = p_data.cpu().detach().numpy()
            s_data, y_data = pred.data.cpu().detach().numpy(), y.data.cpu().detach().numpy()

            if i == 0:
                total_preds = p_data
                total_softmax = s_data
                total_labels = y_data
            else:
                total_preds = np.concatenate((total_preds, p_data))
                total_softmax = np.concatenate((total_softmax, s_data))
                total_labels = np.concatenate((total_labels, y_data))

        if not i == 0 and i % print_freq == 0 or len(loader) - 1 == i:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), loss=losses))

    state = {
        'loss': losses.avg.detach().cpu().numpy().tolist(),
    }

    if return_result:

        if type(total_softmax) is not np.ndarray:
            total_softmax = np.array(total_softmax)
            total_labels = np.array(total_labels)
            total_preds = np.array(total_preds)

        total_softmax = reshape_resulting_array(total_softmax)
        total_preds = total_preds.reshape(-1).squeeze()
        total_labels = total_labels.reshape(-1).squeeze()

        state['softmax_output'] = total_softmax
        state['labels'] = total_labels
        state['results'] = accuracy(total_labels, total_preds, total_softmax)
        state['preds'] = total_preds
    return state


def inference(loader, net, sparsity_ratio, device, base_path, save_result=False):
    pred = []
    label = []
    y_list = []

    net.eval()

    for idx, (X, y) in enumerate(loader):

        pert_out = net(X.to(device))

        if idx == 0:
            # benign_pred = F.softmax(benign_out)
            pred = F.softmax(pert_out)
            # _, tmp_benign_y = benign_pred.data.max(dim=1)
            _, tmp_y = pred.data.max(dim=1)

            # benign_pred = benign_pred.detach().cpu()
            pred = pred.detach().cpu()
            label = tmp_y.detach().cpu()
            y_list = y.detach().cpu()

        else:
            # tmp_benign_pred = F.softmax(benign_out)
            tmp_pred = F.softmax(pert_out)
            # _, tmp_benign_y = tmp_benign_pred.data.max(dim=1)
            _, tmp_y = tmp_pred.data.max(dim=1)

            # benign_pred = torch.cat((benign_pred, tmp_benign_pred.detach().cpu()))
            pred = torch.cat((pred, tmp_pred.detach().cpu()))
            # benign_label = torch.cat((benign_label, tmp_benign_y.detach().cpu()))
            label = torch.cat((label, tmp_y.detach().cpu()))
            y_list = torch.cat((y_list, y.detach().cpu()))

    # benign_score = accuracy(y_list.numpy(), benign_label.numpy(), benign_pred.numpy())
    score = accuracy(y_list.numpy(), label.numpy(), pred.numpy())

    print('Compression Ratio: %d%%' % (sparsity_ratio * 100))
    # print('benign_score: \t', benign_score)
    print('Score: \t', score)

    if save_result:
        file = '{}/{}_{}/'.format(base_path, 'cifar10', 'resnet34')
        save_data((pred, label, score), file,
                  filename='inference_sparsity%d.pkl' % (sparsity_ratio * 100))

    # return benign_pred, pred, benign_label, label, benign_score, adv_score
    return pred, label, score


def training(base_path, device, data_type, data_args, model_name, in_dim, pretrain, optim_name, optim_args, loss_name,
             epochs, sparsity_ratio, custom_pretrained, schedule=None, scheduler_args=None, filename='robust_train', robust_train=False,
             attack_args=None):
    assert epochs > 0, 'epochs must be larger than 0'

    name = 'benign'

    if robust_train:
        name = 'robust'

    tmp_result = {}

    save_point = 10
    trainset, testset, num_classes = getattr(data, data_type)(**data_args)

    net = model_loader(model_name=model_name, pretrain=pretrain, in_features=in_dim, num_class=num_classes)
    print(net)

    if custom_pretrained:
        orig = load_checkpoint(base_path, data_type, model_name, False,
                               filename='benign_train_0')
        net.load_state_dict(orig['state_dict'])
    net.to(device)

    if robust_train:
        adversary = LinfPGDAttack(net, **attack_args)

    optimizer = getattr(optim, optim_name)(net.parameters(), **optim_args)
    criterion = getattr(nn, loss_name)()

    if schedule != None:
        scheduler = getattr(lr_scheduler, schedule)(optimizer, **scheduler_args)

    best_loss = -1
    best_state_dict = {}
    best_test_result = {}
    train_info = {
        'training_info': {
            'base_path': base_path,
            'device': device,
            'data_type': data_type,
            'model_name': model_name,
            'optim_name': optim_name,
            'optim_args': optim_args,
            'loss_name': loss_name,
            'epochs': epochs,
            'sparsity_ratio': sparsity_ratio,
            'pretrain': pretrain,
            'custom_pretrained': custom_pretrained
        },
        'epoch': [],
        'train_loss': [],
        'valid_loss': [],
        'accuracy': [],
        'adv_accuracy': []
    }

    is_best = False

    for epoch in tqdm(range(epochs)):
        if robust_train:  # robust train
            state = robust_train_iter(net, trainset, adversary, criterion, optimizer, epoch, device,
                                      reg_ratio=sparsity_ratio)
            vstate = robust_valid_iter(net, testset, adversary, criterion, epoch, device)

            if schedule != None:
                scheduler.step()
            train_info['epoch'].append(epoch)
            train_info['train_loss'].append(state['loss'])
            train_info['valid_loss'].append(vstate['benign_loss'])
            train_info['accuracy'].append(vstate['benign_acc'])
            train_info['adv_accuracy'].append(vstate['adversarial_acc'])

            tmp_result = {
                'result': vstate,
                'state_dict': state['state_dict'],
                'train_info': train_info
            }

            if vstate['benign_acc'] > best_loss:
                print('best model')
                best_loss = vstate['benign_acc']
                is_best = True

            save_checkpoint(tmp_result, base_path, data_type, model_name, is_best,
                            filename='%s_%s' % (filename, str(100 * sparsity_ratio).replace('.', '_')))

            print(' Epoch: [{0}]\t'
                  'Benign loss {1:.4f}\t'
                  'Benign accuracy {2:.4f}\t'
                  'Adversarial loss {3:.4f}\t'
                  'Adversarial accuracy {4:.4f}\t'.format(
                epoch, vstate['benign_loss'], vstate['benign_acc'], vstate['adversarial_loss'],
                vstate['adversarial_acc']))

            is_best = False

        else:
            state = train_iter(net, trainset, criterion, optimizer, epoch, device, reg_ratio=sparsity_ratio)
            vstate = valid_iter(net, testset, criterion, epoch, device)
            if schedule != None:
                scheduler.step()
            train_info['epoch'].append(epoch)
            train_info['train_loss'].append(state['loss'])
            train_info['accuracy'].append(vstate['results'])

            tmp_result = {
                'result': vstate,
                'state_dict': state['state_dict'],
                'train_info': train_info
            }

            if vstate['results']['accuracy'] > best_loss:
                best_loss = vstate['results']['accuracy']
                is_best = True

            print(' Epoch: [{0}]\t'
                  'Loss {1:.4f}\t'
                  'accuracy {2:.4f}'.format(
                epoch, vstate['loss'], vstate['results']['accuracy']))

            save_checkpoint(tmp_result, base_path, data_type, model_name, is_best,
                            filename='%s_%s' % (filename, str(100 * sparsity_ratio).replace('.', '_')))

            is_best = False

    #     if not epoch == 0 and epoch % save_point == 0:
    #         save_all(tmp_result, base_path, data_type, model_name, '%s_%d.pkl' % (name, sparsity_ratio * 100))
    #
    # save_all(tmp_result, base_path, data_type, model_name, '%s_%d.pkl' % (name, sparsity_ratio * 100))

    return tmp_result
