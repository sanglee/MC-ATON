import torch


def compute_percentile(param, ratio):
    lambda_ = 0
    with torch.no_grad():
        lambda_ = param.abs().view(-1).sort()[0][int(ratio * param.abs().view(-1).shape[0])]
    return lambda_


def get_mask(param, ratio):
    mask = torch.ones_like(param)
    lamb_ = compute_percentile(param, ratio)
    with torch.no_grad():
        mask[torch.abs(param) <= lamb_] = 0
    return mask, lamb_


def compute_module(module, ratio):
    for name, param in module.named_parameters():
        if 'weight' in name:
            mask, lamb_ = get_mask(param, ratio)
    return mask, lamb_


def compute_mask(model, ratio, reg_fc=False):
    named_mask = {}
    named_lamb = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv1d):
            named_mask[name], named_lamb[name] = compute_module(module, ratio)

        if isinstance(module, torch.nn.Linear) and reg_fc:
            named_mask[name], named_lamb[name] = compute_module(module, ratio)
    return named_mask, named_lamb


def regularize_model(named_mask, model):
    last_idx = None
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                p = module.weight
                p.data = p.data * named_mask[name]
                last_idx = named_mask[name]


def structured_idx(layer, sparsity):
    param = layer.weight
    with torch.no_grad():
        p = param.view(param.shape[0], -1)
        p = torch.linalg.norm(p, dim=1)
        lambda_ = p.sort()[0][int(sparsity * p.shape[0])]
        conv_idxs = p.abs() < lambda_


