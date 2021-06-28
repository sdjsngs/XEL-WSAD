"""
model construction function
"""
import  torch
import torch.nn as nn
from  fvcore.common.registry import  Registry
from torch.nn import init

MODEL_REGISTRY=Registry("MODEL")



def weights_init_kaiming(m):
    """
    kaiming init
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    """

    :param m:
    :return:
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def weights_init(m):
    """
    init model weight via model.apply()
    :param m:
    :return:
    """
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.weight)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode="fan_out")
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,0)
        nn.init.constant_(m.bias,0)


def build_model(cfg):
    """
    build model   feature exactor for abnormal detection

    :param cfg:
    :param model_name:
    :return:
    """
    # model_name=cfg.MODEL.MODEL_NAME
    # print("MODEL_REGISTRY", MODEL_REGISTRY.__dict__)
    model = MODEL_REGISTRY.get(cfg.MODEL.MODEL_NAME)(cfg)
    # init model  with xavier
    # model.apply(weights_init)
    model = model.cuda()
    return model



if __name__=="__main__":
    print("model register")
    print("MODEL_REGISTRY", MODEL_REGISTRY.__dict__)