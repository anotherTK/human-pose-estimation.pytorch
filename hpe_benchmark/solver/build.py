
import torch.optim as optim
from .lr_scheduler import WarmupMultiStepLR

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            params,
            # TODO: originally, scale the lr to lr * num_gpu
            lr=lr,
            betas=(0.9, 0.999), 
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            params, 
            lr=lr,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError(
            "The optimizer {} is not defined!".format(cfg.SOLVER.OPTIMIZER))

    return optimizer


def make_lr_scheduler(cfg, optimizer):

    if cfg.SOLVER.SCHEDULER == 'linear':
        w_iters = cfg.SOLVER.WARMUP_ITERS
        w_fac = cfg.SOLVER.WARMUP_FACTOR
        max_iter = cfg.SOLVER.MAX_ITER

        def lr_lambda(iteration):
            updated_lr = 0
            if iteration < w_iters:
                updated_lr = w_fac + (1 - w_fac) * iteration / w_iters
            else:
                updated_lr = 1 - (iteration - w_iters) / (max_iter - w_iters)
            return  updated_lr
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda, 
            last_epoch=-1
        )
    elif cfg.SOLVER.SCHEDULER == 'step':
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise NotImplementedError(
            "The scheduler {} is not defined!".format(cfg.SOLVER.SCHEDULER))


    return scheduler
