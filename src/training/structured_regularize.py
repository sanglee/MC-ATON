import torch
from torch import nn


def get_prun_idx(layer, sparsity, structured=True, criterion='standard', rand_per=0.1):
    assert criterion in ['random', 'large', 'oursr', 'oursl', 'standard'], 'wrong criterion. (random, ours, standard)'

    with torch.no_grad():
        if structured:
            p = layer.weight.data
            p = p.view(p.size(0), -1)
            p = torch.linalg.norm(p, dim=1)
            if criterion == 'random':
                perm = torch.randperm(p.size(0))
                idx = perm[:int(p.size(0) * 0.5)]
                conv_idxs = torch.ones_like(p)
                conv_idxs[idx] = 0
                lambda_ = 0.
            elif criterion == 'oursr':
                lambda_ = p.sort()[0][int(sparsity * rand_per * p.size(0))]
                idx1 = p.sort()[1][:int(p.size(0) * sparsity * (1 - rand_per))]

                perm = torch.randperm(p.size(0) - int(p.size(0) * sparsity * (1 - rand_per)))
                perm = perm[:int(p.size(0) * sparsity * rand_per)]
                idx2 = p.sort()[1][int(p.size(0) * sparsity * (1 - rand_per)):][perm]

                conv_idxs = torch.ones_like(p)
                conv_idxs[torch.cat((idx1, idx2))] = 0
            elif criterion == 'standard':
                lambda_ = p.sort()[0][int(sparsity * p.shape[0])]
                conv_idxs = torch.ones_like(p)
                conv_idxs[p.abs() < lambda_] = 0
            elif criterion == 'large':
                lambda_ = p.sort(descending=True)[0][int(sparsity * p.shape[0])]
                conv_idxs = torch.ones_like(p)
                conv_idxs[p.abs() > lambda_] = 0
            elif criterion == 'oursl':
                lambda_ = p.sort()[0][int(sparsity * rand_per * p.size(0))]
                idx1 = p.sort()[1][:int(p.size(0) * sparsity * (1 - rand_per))]
                idx2 = p.sort()[1][p.size(0) - int(p.size(0) * sparsity * rand_per):-1]
                conv_idxs = torch.ones_like(p)
                conv_idxs[torch.cat((idx1, idx2))] = 0
        else:
            p = layer.weight.abs()
            lambda_ = p.view(-1).sort()[0][int(sparsity * p.view(-1).shape[0])]
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


def structured_lenet_prune(net, sparsity=0., **kwargs):
    is_first = True
    idx_list = []
    lamb_list = []
    for name, module in net.convs.named_modules():
        if isinstance(module, nn.Conv1d):

            if is_first:
                is_first = False
            else:
                conv_batch_prun(module, idxs, structured=True, dim=1)

            idxs, lamb = get_prun_idx(module, sparsity, structured=True, **kwargs)
            idx_list.append(idxs)
            lamb_list.append(lamb)
            conv_batch_prun(module, idxs, structured=True, dim=0)

    for name, module in net.fcs.named_modules():
        if isinstance(module, nn.Linear):
            fc_idxs = idxs
            for i in range(net.conv_out_len - 1):
                fc_idxs = torch.cat((fc_idxs, idxs))

            fc_prun(module, fc_idxs)
            break

    return idx_list, lamb_list


def unstructured_lenet_prune(net, sparsity=0., **kwargs):
    idx_list = []
    lamb_list = []
    for name, module in net.convs.named_modules():
        if isinstance(module, nn.Conv1d):
            idxs, lamb = get_prun_idx(module, sparsity, structured=False, **kwargs)
            conv_batch_prun(module, idxs, structured=False, dim=0)
            idx_list.append(idxs)
            lamb_list.append(lamb)

    for name, module in net.fcs.named_modules():
        if isinstance(module, nn.Linear):
            idxs, lamb = get_prun_idx(module, sparsity, structured=False, **kwargs)
            fc_prun(module, idxs, structured=False)
            break

    return idx_list, lamb_list


def structured_resnet_prune(net, sparsity=0., **kwargs):
    is_first = True
    idx_list = []
    lamb_list = []

    idxs, lamb = get_prun_idx(net.conv1, sparsity, structured=True, **kwargs)
    idx_list.append(idxs)
    lamb_list.append(lamb)
    conv_batch_prun(net.conv1, idxs, structured=True, dim=0)
    conv_batch_prun(net.bn1, idxs, structured=False, dim=0)
    #     with torch.no_grad():
    #         net.bn1.weight[idxs == 0] = 0

    for name, module in net.layer1.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv' in name:
            conv_batch_prun(module, idxs, structured=True, dim=1)
            idxs, lamb = get_prun_idx(module, sparsity, structured=True, **kwargs)
            idx_list.append(idxs)
            lamb_list.append(lamb)
            conv_batch_prun(module, idxs, structured=True, dim=0)

        if isinstance(module, nn.BatchNorm1d) and 'bn' in name:
            conv_batch_prun(module, idxs, structured=False, dim=0)

    channel_idxs = idxs

    for name, module in net.layer2.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv' in name:
            conv_batch_prun(module, idxs, structured=True, dim=1)
            idxs, lamb = get_prun_idx(module, sparsity, structured=True, **kwargs)
            idx_list.append(idxs)
            lamb_list.append(lamb)
            conv_batch_prun(module, idxs, structured=True, dim=0)

        if isinstance(module, nn.BatchNorm1d) and 'bn' in name:
            conv_batch_prun(module, idxs, structured=False, dim=0)

        if isinstance(module, nn.Sequential) and 'downsample' in name:
            conv_batch_prun(module[0], channel_idxs, structured=True, dim=1)
            idxs, lamb = get_prun_idx(module[0], sparsity, structured=True, **kwargs)
            idx_list.append(idxs)
            lamb_list.append(lamb)
            conv_batch_prun(module[0], idxs, structured=True, dim=0)
            conv_batch_prun(module[1], idxs, structured=False, dim=0)

    channel_idxs = idxs
    for name, module in net.layer3.named_modules():

        if isinstance(module, nn.Conv2d) and 'conv' in name:
            conv_batch_prun(module, idxs, structured=True, dim=1)
            idxs, lamb = get_prun_idx(module, sparsity, structured=True, **kwargs)
            idx_list.append(idxs)
            lamb_list.append(lamb)
            conv_batch_prun(module, idxs, structured=True, dim=0)

        if isinstance(module, nn.BatchNorm1d) and 'bn' in name:
            conv_batch_prun(module, idxs, structured=False, dim=0)

        if isinstance(module, nn.Sequential) and 'downsample' in name:
            conv_batch_prun(module[0], channel_idxs, structured=True, dim=1)
            idxs, lamb = get_prun_idx(module[0], sparsity, structured=True, **kwargs)
            idx_list.append(idxs)
            lamb_list.append(lamb)
            conv_batch_prun(module[0], idxs, structured=True, dim=0)
            conv_batch_prun(module[1], idxs, structured=False, dim=0)

    channel_idxs = idxs
    for name, module in net.layer4.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv' in name:
            conv_batch_prun(module, idxs, structured=True, dim=1)
            idxs, lamb = get_prun_idx(module, sparsity, structured=True, **kwargs)
            idx_list.append(idxs)
            lamb_list.append(lamb)
            conv_batch_prun(module, idxs, structured=True, dim=0)

        if isinstance(module, nn.BatchNorm1d) and 'bn' in name:
            conv_batch_prun(module, idxs, structured=False, dim=0)

        if isinstance(module, nn.Sequential) and 'downsample' in name:
            conv_batch_prun(module[0], channel_idxs, structured=True, dim=1)
            idxs, lamb = get_prun_idx(module[0], sparsity, structured=True, **kwargs)
            idx_list.append(idxs)
            lamb_list.append(lamb)
            conv_batch_prun(module[0], idxs, structured=True, dim=0)
            conv_batch_prun(module[1], idxs, structured=False, dim=0)

        #     for name, module in net.fcs.named_modules():
    #         if isinstance(module, nn.Linear):
    #             fc_idxs = idxs
    #             for i in range(net.conv_out_len - 1):
    #                 fc_idxs = torch.cat((fc_idxs, idxs))
    #             fc_prun(module, fc_idxs)
    #             break

    return idx_list, lamb_list
