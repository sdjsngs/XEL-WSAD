"""
misc
"""
import logging
import  torch
import  numpy as np
import  torch
import  psutil
import  datetime
import  torch.nn as nn
import math
import net.utils.logging_tool as logging
from fvcore.nn.flop_count import flop_count
import os
logger=logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    check the loss is NAN ?
    :param loss:
    :return:
    """
    if math.isnan(loss):
        raise RuntimeError(
            "ERROR : Got NAN losses "
        )

def params_count(model):
    """
    compute  mdoel parameters
    :param mdoel:
    :return:
    """
    params_num=np.sum([p.numel() for p in model.parameters()]).item()

    return params_num

def gpu_mem_usage():
    """
      Compute the GPU memory usage for the current device (GB).
      """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 ** 3

def get_flop_stats(model, cfg, is_train):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.

    Returns:
        float: the total number of gflops of the given model.
    """
    rgb_dimension = 1
    # if is_train:
    #     input_tensors = torch.rand(
    #         rgb_dimension,
    #         cfg.DATA.NUM_FRAMES,
    #         cfg.DATA.TRAIN_CROP_SIZE,
    #         cfg.DATA.TRAIN_CROP_SIZE,
    #     )
    # else:
    #     input_tensors = torch.rand(
    #         rgb_dimension,
    #         cfg.DATA.NUM_FRAMES,
    #         cfg.DATA.TEST_CROP_SIZE,
    #         cfg.DATA.TEST_CROP_SIZE,
    #     )
    input_tensors=torch.rand(
        1,16,192,128
    )
    flop_inputs = input_tensors
    for i in range(len(flop_inputs)):
        flop_inputs[i] = flop_inputs[i].unsqueeze(0).cuda(non_blocking=True)

    # If detection is enabled, count flops for one proposal.

    inputs = (flop_inputs,)

    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops

def log_model_info(model,cfg,is_train=True):
    """
    log info mdoel
    :param model:
    :param cfg:
    :param is_train:
    :return:
    """
    # logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info(
        "FLOPs: {:,} GFLOPs".format(get_flop_stats(model, cfg, is_train))
    )



if __name__=="__main__":
    print("misc")