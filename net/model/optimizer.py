"""
optimizer build
"""
import torch
from net.utils.lr_policy import get_lr_at_epoch
def construct_optimizer(model,cfg):

    """
    construct optimizer in
    stochastic gradient descent or ADAM
    set w in decay
    :return:
    """
    # weight_p, bias_p = [], []
    # for name, p in model.named_parameters():
    #     if 'bias' in name:
    #         bias_p += [p]
    #     else:
    #         weight_p += [p]
    # params=[
    #     {"params": weight_p, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
    #     {"params": bias_p, "weight_decay": cfg.SOLVER.BIAS_WEIGHT_DECAY},
    # ]



    if cfg.SOLVER.OPTIMIZING_METHOD=="sgd":
        optim=torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPEMING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD =="adam":
        optim=torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            eps= 1e-10
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD=="adagrad":
        optim=torch.optim.Adagrad(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            lr_decay=0.1,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )
    return optim


def get_epoch_lr(cur_epoch,cfg):
    """
    get cur epoch lr
    :return:
    """

    return get_lr_at_epoch(cfg,cur_epoch)


def set_lr(optimizer,new_lr):
    """
    set new lr in optimizer
    :param optimizer:
    :param new_lr:
    :return:
    """
    for param_group in optimizer.param_groups:
        param_group["lr"]=new_lr


