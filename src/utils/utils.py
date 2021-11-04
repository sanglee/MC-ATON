import os
import pickle
import shutil

import json

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm_notebook as tqdm
import seaborn as sns

def color_palette_to_cmap(color_palette):
    return ListedColormap(color_palette.as_hex())


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


def open_json(file='config.json'):
    with open(file) as config_file:
        jdata = json.load(config_file)

    return jdata


def save_all(data, BASE_PATH, dataset_name, net_name, file_name='data.pkl'):
    directory = '{}/{}_{}/'.format(BASE_PATH, dataset_name, net_name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    save_data(data, directory, file_name)


def get_weights_copy(model, device, weights_path='tmp/weights_temp.pt'):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    torch.save(model.cpu().state_dict(), weights_path)
    model.to(device)
    return torch.load(weights_path)


def accuracy(y_data, p_data, s_data):
    """Computes the precision@k for the specified values of k"""

    f1 = f1_score(y_data, p_data, average='micro')
    acc = accuracy_score(y_data, p_data)
    auc = roc_auc_score(y_data, s_data, multi_class='ovo')
    pre = precision_score(y_data, p_data, average='micro')
    rec = recall_score(y_data, p_data, average='micro')

    return {
        'f1': f1,
        'accuracy': acc,
        'auc_score': auc,
        'precision': pre,
        'recall': rec
    }


def reshape_resulting_array(arr):
    return arr.reshape((-1, arr.shape[-1]))


def get_flat_fts(in_size, fts, device):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.to(device)
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))


def load_checkpoint(base_path, dataset_name, net_name, is_best=False, filename='checkpoint'):
    directory = '{}/{}_{}/'.format(base_path, dataset_name, net_name)
    if is_best:
        file_d = directory + '%s_best' % filename
    else:
        file_d = directory + filename

    print('loading model from %s' % file_d)
    return torch.load('%s.pth.tar' % file_d)


def save_checkpoint(state, base_path, dataset_name, net_name, is_best=False, filename='checkpoint'):
    """Saves checkpoint to disk"""
    directory = '{}/{}_{}/'.format(base_path, dataset_name, net_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_d = directory + filename
    torch.save(state, '%s.pth.tar' % file_d)
    if is_best:
        shutil.copyfile('%s.pth.tar' % file_d, '%s/%s_best.pth.tar' % (directory, filename))


def save_data(state, directory, filename='inference_result.pkl'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file


def plot_grad(g):
    plt.figure(figsize=(10, 8))
    for y_arr, label in zip(g, ['ACCEL', 'CURRENT', 'PEDAL', 'SPEED', 'VOLTAGE']):
        plt.plot(np.arange(0, 100, 1), y_arr, label=label)

    plt.legend()
    plt.show()


def get_grad(net, loader, criterion, device):
    net.eval()
    grads = None

    for i, (X, y) in tqdm(enumerate(loader)):
        X, y = X.to(device), y.to(device)
        X = X.transpose(2, 1)
        X.requires_grad = True
        prob = net(X)
        cost = criterion(prob, y)
        cost.backward()

        grad = X.grad.detach()

        if i == 0:
            grads = grad
        else:
            grads = torch.cat((grads, grad), dim=0)

    return grads.cpu()


def odin_perturb(net, loader, device, epsilon=.1):
    net.eval()
    cost_fn = nn.MSELoss()
    X_tildas = None
    grads = None

    for i, (X, y) in tqdm(enumerate(loader)):
        X, y = X.to(device), y.to(device)
        X = X.transpose(2, 1)
        X.requires_grad = True

        prob = net(X)

        cost = cost_fn(prob, y)
        cost.backward()

        grad = X.grad.detach()
        eta = epsilon * torch.sign(-grad)

        X_tilda = X - eta

        if i == 0:
            X_tildas = X_tilda
            grads = grad
        else:
            X_tildas = torch.cat((X_tildas, X_tilda), dim=0)
            grads = torch.cat((grads, grad), dim=0)

    return X_tildas.cpu(), grads.cpu(), X.cpu(), y.cpu()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
