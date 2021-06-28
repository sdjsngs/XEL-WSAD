"""
inference
save temporal score for each video

"""
import torch
import os
import numpy as np
from net.utils.parser import load_config,parse_args
from net.utils.logging_tool import setup_logging
import torch.nn.functional as F
from net.dataset.loader import construct_loader
from net.model.build import build_model
import  net.utils.logging_tool as logging
import net.model.optimizer as optim
from net.utils.rng_seed import setup_seed
from net.model.losses import get_loss_func
from net.dataset import loader
from net.utils.meter import TestMeter
import net.utils.misc as misc
import net.utils.checkpoint as cu
import  net.utils.tensorboard_vis as Board
import  itertools
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import  scipy.io
# from eval_auc_shanghaitech import eval_auc_roc
from eval_auc_ucf_crime import eval_auc_roc
# logger
logger=logging.get_logger(__name__)

def Tensor2Numpy(input_tensor):
    """
    transform tensor to numpy
    [B,C,H,W] -> [B,H,W,C]
    :param input_tensor:
    :return:
    """
    numpy_x=input_tensor.permute(0,2,3,1).detach().cpu().squeeze(dim=0).numpy()
    return numpy_x


def save_error_npy(npy_name,feature,cfg):
    """
    save error to npy file for auc calculate
    :return:
    """
    save_folder=os.path.join(
        cfg.TEST.SAVE_NPY_PATH,"PRED_TEST_SCORE"
    )
    os.makedirs(save_folder,exist_ok=True)

    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)

    np.save(
        os.path.join(save_folder, npy_name + ".npy"), feature
    )



def infer_epoch(test_loader,model,cfg):
    """

    :param test_loader:
    :param model: model Generator in this project
    :param test_meter:
    :param cfg:
    :return:
    """
    model.eval()

    pred_dict={}

    for cur_iter ,(feature,label,feature_type,video_name) in enumerate(test_loader):
        # batch size =32 for one video  (1,32,4096)
        #custom for  video features
        #[T,4096]
        with torch.no_grad():

            feature = feature.cuda().float().squeeze(dim=0)


            x_1, x_2, x_3, pred_score=model(feature) # time line for 32 value



            save_error_npy(
                video_name[0],pred_score.squeeze().cpu().detach().numpy(),cfg
            )#abnormal_class + "_" +



def infer(cfg):
    """
    infer  func in anomaly detection
    infer in one epoch and save the error to npy
    load model G
    :param cfg:
    :return:
    """

    logging.setup_logging(cfg.OUTPUT_DIR,cfg.TEST_LOGFILE_NAME)

    # build model  attention_fc

    model= build_model(cfg)

    optimizer = optim.construct_optimizer(model, cfg)


    # load checkpoint if exist
    if cu.has_checkpoint(cfg.OUTPUT_DIR,cfg):
        # logger.info("load from last checkpoint")
        last_checkpoint=cu.get_last_checkpoint(cfg.OUTPUT_DIR,cfg)
        _ =cu.load_checkpoint(
            last_checkpoint,model,optimizer
        )

    # elif cfg.TEST.CHECKPOINT_FILE_PATH !="":
    #     logger.info("Load from given checkpoint file")
    #     checkpoint_epoch=cu.load_checkpoint(
    #         cfg.TEST.CHECKPOINT_FILE_PATH,
    #         model_G,
    #         optimizer_G,
    #     )
    # infer_train_data(model,cfg)

    test_loader=loader.construct_loader("test",cfg)
    #
    # # _error_dict=init_error_dict(cfg)
    #
    # # misc.log_model_info(model,cfg)
    #
    infer_epoch(test_loader,model,cfg)



if __name__=="__main__":
    """
    load argpare 
    model 
    data 
    infer and save score   
    
    """

    args=parse_args()
    cfg=load_config(args)


    # # # # #
    infer(cfg)
    eval_auc_roc(cfg)















