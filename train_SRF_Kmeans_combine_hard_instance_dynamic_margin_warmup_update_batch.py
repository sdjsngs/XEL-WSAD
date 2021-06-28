"""
train SRF
batch size =1
train srf with memory bank
dynamic margin
margin_list=[0.6,0.7,0.8,0.9,1.0]
先做warnup 然后执行动态dynamic margin
ucf 总共 100k iteration
shanghai tech 总共10k iteration


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
# from net.utils.K_means_cluster import cluster
from sklearn.cluster import KMeans

# logger
logger=logging.get_logger(__name__)



MARGIN_LIST=[0.5,0.6,0.7,0.8,0.9,1.0]

def k_means_cluster(input_x):
    """

    :param input_x: shape in [T,4096]
    :return:
    """
    array_x=input_x.detach().cpu().numpy()

    cls_kmeans = KMeans(n_clusters=2, random_state=0).fit(array_x)

    pred_label=torch.from_numpy(cls_kmeans.labels_).cuda()
    center_feature=torch.from_numpy(cls_kmeans.cluster_centers_).cuda()


    euc_dis=cal_euclidean(center_feature)

    return pred_label,euc_dis



def cosine_dis(pred_x,pseudo_y):

    cosine_1=torch.cosine_similarity(pred_x,pseudo_y,dim=0)

    cosine_2 = torch.cosine_similarity(pred_x, (pseudo_y-1),dim=0)

    pseudo_y=pseudo_y if cosine_1>cosine_2 else 1-pseudo_y

    return pseudo_y



def cal_euclidean(cluster_center):

    # euc_distance=torch.sqrt(torch.sum(torch.pow((cluster_center[0]-cluster_center[1]),2)))

    euc_distance=torch.dist(cluster_center[0], cluster_center[1], p=2)

    return euc_distance


def update_hard_instance_bank(normal_feature,pred_n_score):
    """

    :param normal_feature: [T,4096]
    :param pred_n_score: [T]
    :return:
    """
    assert  normal_feature.shape[0]==pred_n_score.shape[0]
    normal_score = pred_n_score
    max_n, max_n_index = torch.max(normal_score, dim=0)
    temp_feature=normal_feature[max_n_index.item()].unsqueeze(dim=0)

    # temp_feature=torch.zeros(size=[normal_feature.shape[0],1,4096]).cuda()
    # for i in range (normal_feature.shape[0]):
    #     slect_feature=normal_feature[i][max_n_index[i]].unsqueeze(dim=0)
    #     temp_feature[i]=slect_feature

    return temp_feature



def update_memory_bank_mini_batch(hard_instance_bank,normal_feature,pred_score_normal,normal_index):

    """

    :param hard_instance_bank: shape in [800,4096] for ucf [175,4096] for SH
    :param normal_feature: [T,4096] in SRF
    :param pred_score_normal: [T]
    :param normal_index: tensor shape in [1]
    :return:
    """
    select_normal_feature=update_hard_instance_bank(normal_feature,pred_score_normal)

    index_=normal_index.item()
    hard_instance_bank[index_]=select_normal_feature
    return hard_instance_bank


def train_epoch(
        train_loader,model,optimizer,hard_instance_bank,train_meter,cur_epoch,writer,cfg
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
    # bank_select_flag = False

    for cur_iter,(abnormal_feature,normal_feature,normal_index) in enumerate(train_loader):


        abnormal_feature = abnormal_feature.cuda().float().squeeze(dim=0) # batch size in 1
        normal_feature = normal_feature.cuda().float().squeeze(dim=0)  # batch size in 1

        hard_instance_bank=hard_instance_bank.cuda().float()
        # cal iteration  start from 1
        #total iteration in 100k  each 20k  take a margin

        cur_iteration=(cur_epoch-1)*len(train_loader)+cur_iter+1

        lr=optim.get_epoch_lr(cur_iteration,cfg) # keep lr in 1e-5
        optim.set_lr(optimizer, lr)
        optimizer.zero_grad()


        # warm up  for ucf in 10 epoch for shanghaitech 5 epoch
        # margin_step 1500 and warm up end point 1500 for  shanghai tech
        #  margin step 15000 and 15000 for ucf crime
        margin_step=750
        warm_up_endpoint=750

        if cur_iteration<=warm_up_endpoint:
            dynamic_margin_value=0
            warmup_=True


        else:

            dynamic_margin_value=MARGIN_LIST[(cur_iteration-warm_up_endpoint)//margin_step]
            warmup_=False


        print("cur margin value:{}".format(dynamic_margin_value))


        # perd_score, pseudo_y, euc_dis =model(feature)
        # abnormal feature
        x_1_abnormal, x_2, x_3, pred_score_abnormal=model(abnormal_feature)

        # make cluster and pseudo label
        # x_1 in shape [T,512]
        pseudo_y_abnormal,euc_dis_abnormal=k_means_cluster(x_1_abnormal)  # pseudo_y should change  in cosine smailrly
        pseudo_y_abnormal = cosine_dis(pred_score_abnormal, pseudo_y_abnormal)


        # normal feature
        x_1_normal, x_2, x_3, pred_score_normal=model(normal_feature)

        # make cluster and pseudo label
        # x_1 in shape [T,512]

        pseudo_y_normal,euc_dis_normal=k_means_cluster(x_1_normal)  # pseudo_y should change  in cosine smailrly
        pseudo_y_normal = torch.zeros_like(pseudo_y_normal).cuda()

        # update with mini batch size

        hard_instance_bank=update_memory_bank_mini_batch(
            hard_instance_bank,normal_feature,pred_score_normal,normal_index
        )

        # # make hard loss
        # if cur_epoch==1:
        #     # first epoch  pred hard score=0.5 but
        #     pred_hard_score=torch.ones(size=[cfg.MEMORY_BANK.BANK_SIZE]).cuda()*0.5
        # else:
        #     _,_,_, pred_hard_score=model(hard_instance_bank)

        _,_,_, pred_hard_score = model(hard_instance_bank)

        # normal feature  in shape [T,4096] pred_score_normal [T]

        # temp_select_feature = update_hard_instance_bank(normal_feature, pred_score_normal)


        # if not bank_select_flag: # first choose
        #     hard_instance_bank_update = temp_select_feature
        #     bank_select_flag=True
        # else:
        #     hard_instance_bank_update = torch.cat([hard_instance_bank_update, temp_select_feature], dim=0)


        loss_func = get_loss_func("SRF_LOSS_COMBINE_DYNAMIC_MARGIN_WARMUP_version2") #epoch loss

        # origin SRF losscfg.SOLVER.LOSS_FUNC
        # Lr ：pred_loss mse  for pred score and pseduo_y in video in abnormal
        # lc abnormal : cluster_loss (1/(euc_dis))  normal :(min(upper bound alpha:(1) ,euc_dis))

        total_loss,pred_loss,cluster_loss,hard_instance_hinge_loss,hard_instance_score_loss= loss_func(
            pred_score_abnormal, pseudo_y_abnormal.float(), euc_dis_abnormal,
            pred_score_normal, pseudo_y_normal.float(), euc_dis_normal,
            pred_hard_score,dynamic_margin_value,warmup_
            )


        misc.check_nan_losses(total_loss)
        misc.check_nan_losses(pred_loss)
        misc.check_nan_losses(cluster_loss)
        misc.check_nan_losses(hard_instance_hinge_loss)
        misc.check_nan_losses(hard_instance_score_loss)


        total_loss.backward() #requires_grad=True
        optimizer.step()

        total_loss=total_loss.item()
        pred_loss=pred_loss.item()
        cluster_loss=cluster_loss.item()
        hard_instance_hinge_loss = hard_instance_hinge_loss.item()
        hard_instance_score_loss = hard_instance_score_loss.item()


        train_meter.iter_stop()
        train_meter.update_stats_loss_combine(
            pred_loss,cluster_loss,total_loss,hard_instance_hinge_loss,hard_instance_score_loss, lr, 1 # batch size in 1
        )

        train_meter.log_iter_stats(cur_epoch, cur_iter,"combine")
        train_meter.iter_start()


        # save model in iteraion
        #cfg.TRAIN.CHECKPOINT_PERIOD
        # if cur_iteration>=3000:

        if cu.is_checkpoint_iteration(cur_iteration,cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_iteration, cfg)

        # if cu.is_checkpoint_iteration(cur_iteration=cur_iteration,cfg.TRAIN.CHECKPOINT_PERIOD):
        #     cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_iteration, cfg)


#----------------------------------------------------------------------------------
        writer.add_scalar("lr", lr, cur_iteration )
        writer.add_scalar("pred_loss",pred_loss,cur_iteration)
        writer.add_scalar("cluster_loss", cluster_loss, cur_iteration)
        writer.add_scalar("total_loss", total_loss, cur_iteration)
        writer.add_scalar("hard_instance_hinge_loss", hard_instance_hinge_loss, cur_iteration)
        writer.add_scalar("hard_instance_score_loss", hard_instance_score_loss, cur_iteration)

    train_meter.log_epoch_stats(cur_epoch,"combine")

    train_meter.reset()

    # assert  hard_instance_bank_update.shape[0]==cfg.MEMORY_BANK.BANK_SIZE ,"lack instance in hard instance"
    # cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # logger.info("train {} epoch  finish in {}".format(cur_epoch,cur_time))
    return hard_instance_bank





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
    hard_instance_bank = torch.ones(size=[cfg.MEMORY_BANK.BANK_SIZE,  4096])

    for cur_epoch in range(start_epoch,cfg.SOLVER.MAX_EPOCH+1):
        hard_instance_bank=train_epoch(
            train_loader,model,optimizer,hard_instance_bank,train_meter,cur_epoch,writer,cfg
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


