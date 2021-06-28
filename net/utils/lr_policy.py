"""
lr policy
"""
import  math
from bisect import bisect

def get_lr_at_epoch(cfg,cur_epoch):
    """
    get lr in cur_epoch
    :param cfg:
    :param cur_epoch:
    :return:
    """
    lr=get_lr_func(cfg.SOLVER.LR_POLICY)(cfg,cur_epoch)

    return lr


def lr_func_steps(cfg,cur_epoch):
    """
    stpe lr  policy
    :param cfg:
    :param cur_ecpoh:
    :return:
    """
    ind=int(cur_epoch/10)
    return pow(0.1 ,ind)*cfg.SOLVER.BASE_LR


def lr_func_steps_in_epoch(cfg,cur_epoch):
    """
    stpe lr  policy
    :param cfg:
    :param cur_ecpoh:
    :return:
    """

    ind=bisect(cfg.SOLVER.STEP_EPOCHS,cur_epoch)

    return pow(0.1,ind)*cfg.SOLVER.BASE_LR


def lr_func_steps_in_iteration(cfg,cur_iteration):
    """
    stpe lr  policy
    10K iteration
    initial learning rate of 0.001
    and decrease the learning rate by half at 4K,
    8K and stop at 10K
    :param cfg:
    :param cur_ecpoh:
    :return:
    """

    ind=bisect(cfg.SOLVER.STEP_ITERATIONS,cur_iteration)

    return pow(0.1,ind)*cfg.SOLVER.BASE_LR
def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    # cosine
    return (
        cfg.SOLVER.BASE_LR
        * (math.cos(math.pi * cur_epoch / cfg.SOLVER.MAX_EPOCH) + 1.0)
        * 0.5
    )

def lr_func_keep_linear_decay(cfg,cur_epoch):
    """
    lr keep for cfg.epoch_gate and linear decay to 0
    :param cfg:
    :param cur_ecpoh:
    :return:
    """
    if cur_epoch <cfg.SOLVER.EPOCH_GATE:
        return lr_func_keep(
            cfg, cur_epoch
        )
    else:
        return -(cfg.SOLVER.BASE_LR/(
                cfg.SOLVER.MAX_EPOCH-cfg.SOLVER.EPOCH_GATE
        ))*(cur_epoch-cfg.SOLVER.EPOCH_GATE)+cfg.SOLVER.BASE_LR


def lr_func_keep_cosine(cfg,cur_epoch):
    """
    lr keep for cfg.epoch_gate and linear decay
    :param cfg:
    :param cur_ecpoh:
    :return:

    """
    if cur_epoch <cfg.SOLVER.EPOCH_GATE:
        return lr_func_keep(
            cfg, cur_epoch
        )
    else:
        return (
                cfg.SOLVER.BASE_LR
                * (math.cos(math.pi * (cur_epoch-cfg.SOLVER.EPOCH_GATE) / (cfg.SOLVER.MAX_EPOCH-cfg.SOLVER.EPOCH_GATE)) + 1.0)
                * 0.5
        )


def lr_func_keep(cfg,cur_ecpoh):
    """
    keep lr in base_lr
    :param cfg:
    :param cur_ecpoh:
    :return:
    """
    return cfg.SOLVER.BASE_LR

def get_lr_func(lr_policy):
    """
    give a lr_policy return a learning rate
    :param lr_policy:
    :return:
    """

    lr_func="lr_func_"+lr_policy
    if lr_func not  in globals():
        raise NotImplementedError(
            "Unknown LR policy: {}".format(lr_policy)
        )
    return  globals()[lr_func]


if __name__=="__main__":
    # print(globals())

    print(bisect([10,20],21))