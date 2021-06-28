"""
checkpoints file
"""
import net.utils.logging_tool as logging
import pickle
import os
import  torch
from fvcore.common.file_io import PathManager

logger=logging.get_logger(__name__)


def get_checkpoint_dir(path_to_checkpoint,):
    if not os.path.exists(os.path.join(path_to_checkpoint,"_checkpoints")):
        os.makedirs(os.path.join(path_to_checkpoint,"_checkpoints"))
    return os.path.join(path_to_checkpoint,"_checkpoints")

def get_path_to_checkpoints(path_to_checkpoint,epoch,cfg):

    name="checkpoint_epoch_{:07d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_checkpoint),name)

def get_last_checkpoint(path_to_checkpoint,cfg):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_checkpoint)
    names = PathManager.ls(d) if PathManager.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)

def get_special_checkpoint(path_to_checkpoint,special_epoch,cfg):
    """
    get one special checkpoint
    :param path_to_checkpoint:
    :return:
    """

    d = get_checkpoint_dir(path_to_checkpoint)

    names = PathManager.ls(d) if PathManager.exists(d) else []
    special_name = "checkpoint_epoch_{:07d}.pyth".format(special_epoch)
    names = [f for f in names if special_name in f]
    name=names[0]
    print("load mode in special epoch : {}".format(os.path.join(d, name)))
    # logger.info("load mode in special epoch : {}".format(os.path.join(d, name)))
    return os.path.join(d, name)


def has_checkpoint(path_to_checkpoint,cfg):
    """
    check if checkpoint exist
    :param path_to_checkpoint:
    :return:
    """
    d=get_checkpoint_dir(path_to_checkpoint)
    files=PathManager.ls(d) if PathManager.exists(d) else []

    return any("checkpoint" in f for f in files)

def checkpoint_num(path_to_checkpoint,cfg):
    """
    return checkpoint num for model with model name
    :param path_to_checkpoint:
    :param model_name:
    :param cfg:
    :return:
    """
    d = get_checkpoint_dir(path_to_checkpoint)
    files = PathManager.ls(d) if PathManager.exists(d) else []

    return len(files)

def is_checkpoint_epoch(cur_epoch,checkpoint_period):
    """
    determine if a checkpoint should be saved in cur_epoch
    :param cur_epoch:
    :param checkpoint_period:
    :return:
    """
    return (cur_epoch+1)%checkpoint_period==0

def is_checkpoint_iteration(cur_iteration,checkpoint_period):
    """
    determine if a checkpoint should be saved in cur_epoch
    :param cur_epoch:
    :param checkpoint_period:
    :return:
    """
    return (cur_iteration)%checkpoint_period==0

def save_checkpoint(path_to_checkpoint,model,optimizer,epoch,cfg):
    """
    save a checkpoint
    :param path_to_checkpoint:
    :param mdoel:
    :param mdoel: G or D
    :param optimizer:
    :param epoch:
    :param cfg:
    :return:
    """
    logger.info("save checkpoint in epoch {}".format(epoch))

    PathManager.mkdirs(get_checkpoint_dir(path_to_checkpoint))
    sd=model.state_dict()

    #Recode the state
    checkpoint={
        "epoch":epoch,
        "model_state":sd,
        "optimizer_state":optimizer.state_dict(),
        "cfg":cfg.dump()
    }

    checkpoint_path=get_path_to_checkpoints(path_to_checkpoint,epoch,cfg,)
    # if (epoch+1)%10==0 or (epoch+1)==cfg.SOLVER.MAX_EPOCH:
    with PathManager.open(checkpoint_path,"wb") as f:
        torch.save(checkpoint,f)
    return checkpoint_path



def load_checkpoint(path_to_checkpoint,model,optimizer):
    """
    load checkpoint
    :return:
    """
    assert  PathManager.exists(path_to_checkpoint), "checkpoint {}".format(path_to_checkpoint)
    # load checkpoint on cpu
    with PathManager.open(path_to_checkpoint,"rb") as f:
        checkpoint=torch.load(f,map_location="cpu")
        print("checkpoint {} is load ".format(path_to_checkpoint.split("/")[-1]))
    ms=model
    ms.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint.keys():
        epoch=checkpoint["epoch"]
    else:
        epoch=-1
    return epoch


def load_pretrain_model(model,pre_train_path):
    """
    load pre-train model
    and  custom the layer in model
    :return:
    """

    net=model
    net_state=net.state_dict()
    # load pre-train weight
    model_dict=torch.load(pre_train_path)

    #model_dict={k:v for k,v in model_dict.items() for k in net_state}

    net.load_state_dict(model_dict)
    return net


if __name__=="__main__":
    print("checkpoint")
