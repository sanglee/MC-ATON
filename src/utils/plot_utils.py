#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/08/23 6:20 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : plot_utils.py
# @Software  : PyCharm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision.utils import make_grid

import cv2

def plot_3channel():
    img = cv2.imread('./data/home.jpg')
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def plot_loss(losses, names, figsize=(10, 10)):
    sns.set_palette('muted', len(losses))
    plt.figure(figsize=figsize)
    plt.title('Loss Plot')
    for l, n in zip(losses, names):
        plt.plot(l, '-o', label=n)
    plt.xlabel('EPOCH')
    plt.ylabel('loss')
    plt.legend()

    plt.show()


def plot_grid(imgs, nrow=4, figsize=(16, 16), reg=True, filename=None):
    img = make_grid(imgs, nrow=nrow)
    if reg:
        img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    if not filename == None:
        plt.savefig('assets/%s'%filename)



def img_plot_sample(imgs, size=(4, 4), figsize=(16, 16)):
    fig, ax = plt.subplots(size[0], size[1], figsize=figsize)
    for i in range(size[0] * size[1]):
        ax[i // size[0]][i % size[1]].imshow(imgs[i].transpose((1, 2, 0)), interpolation='nearest')
    plt.show()


def plot_two_axis(d1, d2, x, d1_name, d2_name, c1=sns.color_palette('muted')[0], c2=sns.color_palette('muted')[1]):
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 8))
    #     ax1.grid(False)

    ax1.set_title('%s and %s per sparsity' % (d1_name, d2_name))
    ax2 = ax1.twinx()
    ax2.grid(False)

    ax1.plot(x, d1, '-o', c=c1, label=d1_name)
    ax2.plot(x, d2, '-o', c=c2, label=d2_name)

    ax1.set_xlabel('sparsity', fontsize=18)
    ax1.set_ylabel(d1_name, color=c1, fontsize=18)
    ax2.set_ylabel(d2_name, color=c2, fontsize=18)
    plt.show()


def multiple_timepoint_plot(x, ls, fil_ls=None, title='MSELoss', xlabel='Timepoint', ylabel='MSELoss', ylim=(0.5, 6.2)):
    plt.figure(figsize=(10, 8))

    for i, l in enumerate(ls):
        plt.plot(l, 'o-', label='sparsity %d%%' % int(x[i] * 100), c=sns.color_palette('muted')[i])
        if fil_ls is not None:
            plt.fill_between(np.arange(0, 25, 1), l - fil_ls[i], l + fil_ls[i], alpha=0.2,
                             label='variance at sparsity %d%%' % int(x[i] * 100))

    plt.title('BERT MSELoss', fontsize=20)
    plt.legend(fontsize=13)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.ylim(ylim)


def plot_sparsity(d1, d2, sparsity, d1_name='variance', d2_name='capacity', c1=sns.color_palette('muted')[0],
                  c2=sns.color_palette('muted')[1]):
    fig, ax1 = plt.subplots(figsize=(10, 8))

    ax1.set_title('%s and %s per sparsity' % (d1_name, d2_name))
    ax2 = ax1.twinx()

    ax1.plot(sparsity, d1, '-o', c=c1, label=d1_name)
    ax2.plot(sparsity, d2, '-o', c=c2, label=d2_name)

    ax1.set_xlabel('sparsity', fontsize=18)
    ax1.set_ylabel(d1_name, color=c1, fontsize=18)
    ax2.set_ylabel(d2_name, color=c2, fontsize=18)
    plt.show()


def plot_classes(trainset_var_mean_per_sparsity, sparsity, num_class=10):
    fig, axs = plt.subplots(int(num_class / 5), 5, figsize=(30, 15))

    for num in np.arange(0, num_class):
        ax = axs[int(num / 5), int(num % 5)]

        var = trainset_var_mean_per_sparsity['var'][:, num]
        mean = trainset_var_mean_per_sparsity['mean'][:, num]
        f1 = trainset_var_mean_per_sparsity['f1']
        acc = trainset_var_mean_per_sparsity['acc']

        ax.set_ylim((0.95, 1.01))

        ax.plot(sparsity, mean, 'o-', label='mean')
        ax.fill_between(sparsity, mean - var, mean + var, alpha=0.2, label='variance')
        ax.plot(sparsity, acc, 'o-', label='accuracy')

        ax.set(xlabel='sparsity', ylabel='softmax')
        ax.set_title('class %d' % num)
    plt.legend(fontsize=15)
