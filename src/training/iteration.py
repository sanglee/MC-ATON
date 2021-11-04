import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

import training
from training.bert_pruning import bert_prune
from training.perturb import epsilon_ball_noise
from training.regularize import regularize_model, compute_mask
from utils import AverageMeter, reshape_resulting_array, accuracy

BASE_PATH=''

def iteration(model, loader, criterion, name='L0LeNet', isTrain=True, isConstrain=False, sparsity=0., print_freq=100,
              optimizer=None, epoch=0, device=3, cls=False, transpose=False, regularizer='structured_lenet_prune',
              reg_criterion='standard', rand_per=0.5, isBert=False, stability_check=False, robustTrain=True,
              epsilon=0.1):
    assert reg_criterion in ['random', 'large', 'oursr', 'oursl',
                             'standard'], "wrong regularization criterion. ('random', 'large', 'oursr', 'oursl', 'standard')"

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if robustTrain:
        mse_l = AverageMeter()
        mse_hat_l = AverageMeter()

    if stability_check and sparsity > 0.:
        prune_stability = AverageMeter()
    end = time.time()

    total_softmax = []
    total_labels = []
    total_preds = []

    model.train() if isTrain else model.eval()

    for i, (X, y) in enumerate(loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            X, y = X.to('cuda:%d' % device), y.to('cuda:%d' % device)
        if transpose:
            X = X.transpose(2, 1)

        pred = model(X)

        if robustTrain:
            X_hat = epsilon_ball_noise(X, epsilon=epsilon)
            pred_hat = model(X_hat.to('cuda:%d' % device))
            loss, mse, mse_hat = criterion(pred, pred_hat, y)
            mse_l.update(mse.data, X.size(0))
            mse_hat_l.update(mse_hat.data, X.size(0))
        else:
            loss = criterion(pred, y)
        losses.update(loss.data, X.size(0))

        if isTrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if isConstrain:
            # layers = model.layers
            # for layer in layers:
            #     layer.clamp_params()

            if stability_check and sparsity > 0.:
                model.eval()
                test_p = model(X)
                l_pre = criterion(test_p, y)
                model.train()

            if isBert:
                bert_prune(model, sparsity)
            else:
                getattr(training, regularizer)(model, sparsity=sparsity, criterion=reg_criterion, rand_per=rand_per)

            if stability_check and sparsity > 0.:
                model.eval()
                test_p = model(X)
                l_post = criterion(test_p, y)
                model.train()

                prune_stability.update(1 - (torch.abs(l_pre.data - l_post.data) / l_pre.data))

        # get metric
        if cls:
            pred = F.softmax(pred)
            _, p_data = pred.data.max(dim=1)
            p_data = p_data.cpu().detach().numpy()
            total_preds.append(p_data)

        s_data, y_data = pred.data.cpu().detach().numpy(), y.data.cpu().detach().numpy()
        total_softmax.append(s_data)
        total_labels.append(y_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            if stability_check and sparsity > 0.:
                if robustTrain:
                    print(' Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'prune stability {stability.val:.3f} ({stability.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MSE {mse.avg:.4f} MSE_hat {mse_hat.avg:.4f}\t'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        stability=prune_stability, loss=losses, mse=mse_l, mse_hat=mse_hat_l))
                else:
                    print(' Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'prune stability {stability.val:.3f} ({stability.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        stability=prune_stability, loss=losses))

            else:
                if robustTrain:
                    print(' Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MSE {mse.avg:.4f} MSE_hat {mse_hat.avg:.4f}\t'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        loss=losses, mse=mse_l, mse_hat=mse_hat_l))
                else:
                    print(' Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        loss=losses))

    if isConstrain:
        # layers = model.layers
        # for layer in layers:
        #     layer.clamp_params()
        if isBert:
            bert_prune(model, sparsity)
        else:
            getattr(training, regularizer)(model, sparsity=sparsity, criterion='standard', rand_per=rand_per)

    if cls:
        total_softmax = reshape_resulting_array(total_softmax)
        total_labels = np.array(total_labels).reshape(-1).squeeze()
    else:
        if type(total_softmax) is not np.ndarray:
            total_softmax = np.array(total_softmax)
            total_labels = np.array(total_labels)

    state = {
        'name': name,
        'epoch': epoch + 1,
        'state_dict': deepcopy(model).cpu().state_dict(),
        'loss': losses.avg.detach().cpu().numpy().tolist(),
        'softmax_output': total_softmax,
        'labels': total_labels,
    }

    if cls:
        total_preds = np.array(total_preds).reshape(-1).squeeze()
        state['results'] = accuracy(total_labels, total_preds, total_softmax)
        state['preds'] = total_preds

    if isTrain:
        state['optimizer'] = deepcopy(optimizer).state_dict()

    if stability_check and sparsity > 0.:
        state['prune_stability'] = prune_stability.avg.detach().cpu().numpy()

    return state


def iteration_others(model, loader, criterion, name='resnet18', isTrain=True, isConstrain=False, sparsity=0,
                     print_freq=100, optimizer=None, epoch=0, device=3, cls=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    total_softmax = []
    total_labels = []
    total_preds = []

    model.train() if isTrain else model.eval()

    for i, (X, y) in enumerate(loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            X, y = X.to('cuda:%d' % device), y.to('cuda:%d' % device)

        pred = model(X)
        loss = criterion(pred, y)
        losses.update(loss.data, X.size(0))

        if isTrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if isConstrain and isTrain:
            mask, lamb = compute_mask(model, ratio=sparsity)
            regularize_model(mask, model)

        # get metric
        if cls:
            pred = F.softmax(pred, dim=1)

        _, p_data = pred.data.max(dim=1)
        s_data, y_data = pred.data.cpu().detach().numpy().tolist(), y.data.cpu().detach().numpy().tolist()
        p_data = p_data.cpu().detach().numpy().tolist()

        total_softmax.append(s_data)
        total_labels.append(y_data)
        total_preds.append(p_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    if type(total_softmax) is not np.ndarray:
        total_softmax = np.array(total_softmax)
        total_labels = np.array(total_labels)
        total_preds = np.array(total_preds)

    total_softmax = reshape_resulting_array(total_softmax)
    total_labels = total_labels.reshape(-1).squeeze()
    total_preds = total_preds.reshape(-1).squeeze()

    print(total_softmax.shape, total_labels.shape, total_preds.shape)

    result = accuracy(total_labels, total_preds, total_softmax)

    # if save_state:
    state = {
        'name': name,
        'epoch': epoch + 1,
        'state_dict': deepcopy(model).cpu().state_dict(),
        'results': result,
        'loss': losses.avg.detach().cpu().numpy(),
        'softmax_output': total_softmax,
        'labels': total_labels,
        'preds': total_preds
    }
    # if isTrain:
    #     state['optimizer'] = deepcopy(optimizer).state_dict()

    return state
