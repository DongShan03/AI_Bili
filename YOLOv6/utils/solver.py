import os
import math
import torch
import torch.nn as nn

def build_optimizer(cfg, model):
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)

    assert cfg.solver.optim == 'SGD' or "Adam", "ERROR: Optimizer {} is not supported".format(cfg.solver.optim)
    if cfg.solver.optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))

    optimizer.add_param_group({"params": g_w, "weight_decay": cfg.solver.weight_decay})
    optimizer.add_param_group({"params": g_b})

    del g_bnw, g_w, g_b

    return optimizer

def build_lr_scheduler(cfg, optimizer, epochs):
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((x - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        print("ERROR: Scheduler {} is not supported".format(cfg.solver.lr_scheduler))
        exit(-1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler
