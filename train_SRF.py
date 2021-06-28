"""
train SRF
batch size =1

"""
import torch
import numpy as np
import time
from net.utils.parser import load_config,parse_args
from net.model.build import build_model
import  net.utils.logging_tool as logging
import net.model.optimizer as optim
from net.utils.rng_seed import setup_seed
from net.model.losses import get_loss_func
from net.dataset import loader
from net.utils.meter_SRF import TrainMeter
import net.utils.misc as misc
import net.utils.checkpoint as cu
import net.utils.tensorboard_vis as Board
# from kmeans_pytorch import kmeans
from net.utils.K_means_cluster import cluster
# logger
logger=logging.get_logger(__name__)


def k_means_cluster(input_x):
    # kmeans for x_1
    # cluster to 2 center
    # cluster_ids_x, cluster_centers = kmeans(
    #     X=input_x, num_clusters=2, distance='euclidean', device=torch.device('cuda')
    # )
    cluster_ids_x, cluster_centers = cluster(input_x)

    cluster_ids_x=cluster_ids_x.cuda()
    cluster_centers=cluster_centers.cuda()

    euc_dis=cal_euclidean(cluster_centers)

    return cluster_ids_x,euc_dis



def cosine_dis(pred_x,pseudo_y):

    cosine_1=torch.cosine_similarity(pred_x,pseudo_y,dim=0)

    cosine_2 = torch.cosine_similarity(pred_x, (pseudo_y-1),dim=0)

    pseudo_y=pseudo_y if cosine_1>cosine_2 else 1-pseudo_y

    return pseudo_y



def cal_euclidean(cluster_center):

    # euc_distance=torch.sqrt(torch.sum(torch.pow((cluster_center[0]-cluster_center[1]),2)))

    euc_distance=torch.dist(cluster_center[0], cluster_center[1], p=2)

    return euc_distance





def train_epoch(
        train_loader,model,optimizer,train_meter,cur_epoch,writer,cfg
):
    """
    :param train_loader:
    :param model:
    :param optimizer:
    :param train_meter:
    :param cur_epoch:
    :param writer:
    :param cfg:
    :return:
    train multi-instance
    memory bank collect
    loss backward

    """

    model.train()
    train_meter.iter_start()

    for cur_iter,(feature,label,flag) in enumerate(train_loader):


        feature = feature.cuda().float().squeeze(dim=0) # batch size in 1

        # cal iteration  start from 1

        cur_iteration=(cur_epoch-1)*len(train_loader)+cur_iter+1
        lr=optim.get_epoch_lr(cur_iteration,cfg) # keep lr in 1e-5
        optim.set_lr(optimizer, lr)
        optimizer.zero_grad()


        # perd_score, pseudo_y, euc_dis =model(feature)

        x_1, x_2, x_3, pred_score=model(feature)

        # make cluster and pseudo label
        # x_1 in shape [T,512]
        pseudo_y,euc_dis=k_means_cluster(x_1)  # pseudo_y should change  in cosine smailrly


        if flag[0] in ["Abnormal"]:
            pseudo_y=cosine_dis(pred_score,pseudo_y)
        elif flag[0] in ["Normal"]:
            pseudo_y = torch.zeros_like(pseudo_y).cuda()
        else: raise NotImplementedError(
            "Not supported is_abnormal {}".format(flag[0])
        )



        loss_func = get_loss_func("SRF_LOSS") # origin SRF loss
        # Lr ：pred_loss mse  for pred score and pseduo_y in video in abnormal
        # lc abnormal : cluster_loss (1/(euc_dis))  normal :(min(upper bound alpha:(1) ,euc_dis))

        total_loss,pred_loss,cluster_loss= loss_func(
            pred_score, pseudo_y.float(), euc_dis,flag[0]
            )


        misc.check_nan_losses(total_loss)
        misc.check_nan_losses(pred_loss)
        misc.check_nan_losses(cluster_loss)

        # total_loss=total_loss.float()
        # print("CHECK total loss type",type(total_loss))

        total_loss.backward() #requires_grad=True
        optimizer.step()

        total_loss=total_loss.item()
        pred_loss=pred_loss.item()
        cluster_loss=cluster_loss.item()



        train_meter.iter_stop()
        train_meter.update_stats_origin(
            pred_loss,cluster_loss,total_loss, lr, feature.size(0) # batch size in 1
        )

        train_meter.log_iter_stats(cur_epoch, cur_iter,"origin")
        train_meter.iter_start()


        # save model in iteraion
        #cfg.TRAIN.CHECKPOINT_PERIOD
        if cu.is_checkpoint_iteration(cur_iteration,cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_iteration, cfg)

        # if cu.is_checkpoint_iteration(cur_iteration=cur_iteration,cfg.TRAIN.CHECKPOINT_PERIOD):
        #     cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_iteration, cfg)


#----------------------------------------------------------------------------------
        writer.add_scalar("lr", lr, cur_iteration )
        writer.add_scalar("pred_loss",pred_loss,cur_iteration)
        writer.add_scalar("cluster_loss", cluster_loss, cur_iteration)
        writer.add_scalar("total_loss", total_loss, cur_iteration)

    train_meter.log_epoch_stats(cur_epoch,"origin")

    train_meter.reset()
    # cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # logger.info("train {} epoch  finish in {}".format(cur_epoch,cur_time))




def train(cfg):
    """
    train func in anomaly detection

    :param cfg:
    :return:
    """

    logging.setup_logging(cfg.OUTPUT_DIR,cfg.TRAIN_LOGFILE_NAME)
    logger.info("train with config")
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logger.info("train time start from {}".format(cur_time))

    # build model
    model = build_model(cfg) #SRF fc

    optimizer = optim.construct_optimizer(model,cfg) # adam

    # # load checkpoint if exist
    # if cu.has_checkpoint(cfg.OUTPUT_DIR,cfg):
    #     logger.info("load from last checkpoint")
    #
    #     last_checkpoint=cu.get_last_checkpoint(cfg.OUTPUT_DIR,cfg)
    #     last_epoch =cu.load_checkpoint(
    #         last_checkpoint,model,optimizer
    #     )
    #
    #     start_epoch=last_epoch+1

    # elif cfg.TRAIN.CHECKPOINT_FILE_PATH !="":
    #     logger.info("Load from given checkpoint file")
    #     checkpoint_epoch=cu.load_checkpoint(
    #         cfg.TRAIN.CHECKPOINT_FILE_PATH,
    #         model,
    #         optimizer,
    #     )
    #     start_epoch = checkpoint_epoch + 1
    # else:

    start_epoch=1


    train_loader=loader.construct_loader("train",cfg)

    train_meter=TrainMeter(len(train_loader),cfg)

    logger.info("Start epoch {}".format(start_epoch))

    writer=Board.init_summary_writer(cfg.OUTPUT_DIR,cur_time)



    # # cal max epoch
    #total iteration in  100k and save model in each 500 iteration
    if cfg.SOLVER.MAX_ITERATION % len(train_loader) == 0:
        max_epoch=cfg.SOLVER.MAX_ITERATION//len(train_loader)
    else:
        max_epoch = cfg.SOLVER.MAX_ITERATION // len(train_loader)+1
    cfg.SOLVER.MAX_EPOCH=max_epoch

    #------------------------------------------------------------------
    # total train in 100k iteration
    logger.info(" cfg.SOLVER.MAX_ITERATION {}".format(cfg.SOLVER.MAX_ITERATION))
    logger.info(" len(train_loader) {}".format(len(train_loader)))
    logger.info(" max_epoch {}".format(cfg.SOLVER.MAX_EPOCH))
    #------------------------------------------------------------------

    for cur_epoch in range(start_epoch,cfg.SOLVER.MAX_EPOCH+1):
        train_epoch(
            train_loader,model,optimizer,train_meter,cur_epoch,writer,cfg
        )

        #
        # # save checkpoint
        # if cu.is_checkpoint_epoch(cur_epoch,cfg.TRAIN.CHECKPOINT_PERIOD):
        #     cu.save_checkpoint(cfg.OUTPUT_DIR, model,optimizer,cur_epoch,cfg)

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logger.info("train end in {}".format(cur_time))
    writer.close()

if __name__=="__main__":

    """
    load argpare 
    model 
    data 
    train  
    save model and tensor board 
    """
    args=parse_args()
    cfg=load_config(args)
    setup_seed(cfg.RNG_SEED)
    train(cfg)


