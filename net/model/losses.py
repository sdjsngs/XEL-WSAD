"""
loss function
img l2 loss
flow l1 loss
GAN loss

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




def L1_loss(img_pred,img):
    l1_loss=nn.L1Loss()
    loss=l1_loss(img_pred,img)
    return loss

def L2_loss(pred_score,label):
    l2_loss=nn.MSELoss(reduction='mean')
    loss=l2_loss(pred_score,label)
    return loss

def BCE_loss(img_pred,img_label):
    bce_loss=nn.BCELoss()
    loss=bce_loss(img_pred,img_label)
    return loss


def hinge_loss(abnormal_score,normal_score):
    """
    hinge loss
    loss=max(0,1-max(abnormal)+max(normal))
    :param abnormal_score: [B,32,1]
    :param normal_score: [B,32,1]
    :return:
    """

    abnormal_score=abnormal_score.squeeze()
    normal_score=normal_score.squeeze()

    max_a_value,max_a_index=torch.max(abnormal_score,dim=-1)  # batch_size
    max_n_value,max_n_index=torch.max(normal_score,dim=-1)


    margin_1=torch.ones_like(max_a_value)
    # margin_0=torch.zeros_like(max_a)

    # margin_loss = nn.MarginRankingLoss()
    #
    # h_loss=margin_loss(max_a,max_n,margin_1)

    h_loss=F.relu((margin_1 - max_a_value + max_n_value))


    return h_loss,max_a_index,max_n_index



def T_1_loss(abnormal_score):
    """
    smooth loss

    :param abnormal_score:
    :return:
    """
    abnormal_score=abnormal_score.squeeze(dim=-1)

    p_score=abnormal_score[:,:-1]
    l_score=abnormal_score[:,1:]

    # p_score=abnormal_score[:-1]
    # l_score=abnormal_score[1:]

    # do l2 or l1
    # l1_loss=torch.sum(
    #     torch.abs(p_score-l_score)
    # )

    l2_loss=torch.sum(
            torch.pow(p_score - l_score, 2), dim=-1
        )


    return l2_loss

def T_2_loss(abnormal_score):
    """
    sparsity loss
    :param abnormal_score:[30,32,1]
    :return: shape  [30]
    """
    loss_value=torch.sum(abnormal_score.squeeze(dim=-1),dim=-1)
    return loss_value



def combine_loss(abnormal_score,normal_score):
    """
    combine loss
    abnormal score shape in [b,t,1]
    normal score shape in [b,t,1]
    hyp= 8X10^-5
    :return:
    """
    h_loss,max_a_index,max_n_index=hinge_loss(abnormal_score,normal_score)
    smooth_loss=T_1_loss(abnormal_score)
    sparsity_loss=T_2_loss(abnormal_score)

    hyp=0.00008

    combine_loss=torch.mean(h_loss+hyp*smooth_loss+hyp*sparsity_loss)

    return combine_loss,h_loss.mean(),smooth_loss.mean(),sparsity_loss.mean(),max_a_index,max_n_index


def hard_sample_loss(abnormal_score,hard_instance_score):
    """

    :param abnormal_score: [30,32,1]
    :param hard_instance_score: [800,1,1]
    :return:
    """
    abnormal_size=abnormal_score.shape[0]
    memory_size=hard_instance_score.shape[0]

    abnormal_score = abnormal_score.squeeze()


    max_a, max_a_index = torch.max(abnormal_score, dim=1) # (30,1)


    max_a_repeat=max_a.unsqueeze(dim=1).repeat(1,memory_size).permute(1,0).flatten() # shape in [memory size ,30 ]

    hard_instance_score=hard_instance_score.squeeze(dim=-1).repeat(1,abnormal_size).flatten()
    margin_1=torch.ones_like(max_a_repeat)

    hard_loss=torch.mean(
        F.relu((margin_1 - max_a_repeat + hard_instance_score))
    )



    return hard_loss

def hard_sample_loss_remove_one(abnormal_score,hard_instance_score):
    """

    :param abnormal_score: [30,32,1]
    :param hard_instance_score: [800,1,1]
    :return:
    """
    abnormal_size=abnormal_score.shape[0]
    memory_size=hard_instance_score.shape[0]

    abnormal_score = abnormal_score.squeeze()


    max_a, max_a_index = torch.max(abnormal_score, dim=1) # (30,1)


    max_a_repeat=max_a.unsqueeze(dim=1).repeat(1,memory_size).permute(1,0).flatten() # shape in [memory size ,30 ]

    hard_instance_score=hard_instance_score.squeeze(dim=-1).repeat(1,abnormal_size).flatten()
    # margin_1=torch.ones_like(max_a_repeat)

    hard_loss=torch.mean(
        F.relu((max_a_repeat - hard_instance_score))
    )



    return hard_loss


def combine_loss_hard_sample(abnormal_score,normal_score,hard_instance_score):
    """
    combine loss
    abnormal score shape in [B,T,1]
    normal score shape in [B,T,1]
    hard_instance_score in [memory_size ,1,1 ]
    hyp= 8X10^-5
    :return:
    """
    # abnormal score and
    h_loss,max_a_index,max_n_index=hinge_loss(abnormal_score,normal_score)
    smooth_loss=T_1_loss(abnormal_score)
    sparsity_loss=T_2_loss(abnormal_score)
    hard_loss=hard_sample_loss_remove_one(abnormal_score,hard_instance_score)
    # min the hard score
    hard_min_score=torch.mean(hard_instance_score.squeeze())
    hyp=0.00008

    combine_loss=torch.mean(h_loss+hyp*smooth_loss+hyp*sparsity_loss)+hard_loss+hard_min_score

    return combine_loss,h_loss.mean(),smooth_loss.mean(),sparsity_loss.mean(),hard_loss,hard_min_score #,max_a_index,max_n_index


def combine_loss_1_hard_sample(abnormal_score,normal_score,hard_instance_score):
    """
    combine loss
    abnormal score
    normal score
    hyp= 8X10^-5
    plus loss 1
    :return:
    """
    # abnormal score and
    h_loss,max_a_index,max_n_index=hinge_loss(abnormal_score,normal_score)
    smooth_loss=T_1_loss(abnormal_score)
    sparsity_loss=T_2_loss(abnormal_score)
    hard_loss=hard_sample_loss(abnormal_score,hard_instance_score)

    hyp=0.00008

    combine_loss=torch.mean(h_loss+hyp*smooth_loss+hyp*sparsity_loss)+hard_loss

    return combine_loss,h_loss.mean(),smooth_loss.mean(),sparsity_loss.mean(),hard_loss#,max_a_index,max_n_index





def combine_loss_2_hard_sample(abnormal_score,normal_score,hard_instance_score):
    """
    combine loss
    abnormal score
    normal score
    hyp= 8X10^-5
    plus loss 2
    :return:
    """
    # abnormal score and
    h_loss,max_a_index,max_n_index=hinge_loss(abnormal_score,normal_score)
    smooth_loss=T_1_loss(abnormal_score)
    sparsity_loss=T_2_loss(abnormal_score)
    # hard_loss=hard_sample_loss(abnormal_score,hard_instance_score)
    # min the hard score
    hard_min_score = torch.mean(hard_instance_score.squeeze())
    hyp=0.00008

    combine_loss=torch.mean(h_loss+hyp*smooth_loss+hyp*sparsity_loss)+hard_min_score

    return combine_loss,h_loss.mean(),smooth_loss.mean(),sparsity_loss.mean(),hard_min_score#,max_a_index,max_n_index




class RegularizedLoss(torch.nn.Module):
    """
    ||w|| regular weight
    """
    def __init__(self, model, lambdas=0.001):
        super(RegularizedLoss, self).__init__()
        self.lambdas = lambdas
        self.model = model


    def forward(self, y_pred, y_true):
        # loss
        # Our loss is defined with respect to l2 regularization, as used in the original keras code
        fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.fc1.parameters()]))
        fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.fc2.parameters()]))
        fc3_params = torch.cat(tuple([x.view(-1) for x in self.model.fc3.parameters()]))

        l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
        l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
        l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)
        regular_loss=l1_regularization + l2_regularization + l3_regularization

        return regular_loss


def SRF_loss(pred_score,pseudo_y,euc_dis,video_label="Abnormal"):
    """
    loss in A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels
    :param pred_score:
    :param pseudo_y:
    :param euc_dis:
    :param video_label: Abnormal or Normal
    :return:
    """
    upper_bound_alpha=torch.tensor([1.0]).cuda()
    hyp=0.05

    # Lr  mse loss in pred_score and
    L_r=L2_loss(pred_score,pseudo_y)

    if video_label in ["Normal"]:
        L_c=euc_dis if euc_dis<upper_bound_alpha else upper_bound_alpha

    elif video_label in ["Abnormal"]:
        L_c  = 1.0/(euc_dis+1e-8)

    total_loss=L_r+hyp*L_c


    return total_loss,L_r,L_c

def SRF_hard_hinge_loss(abnormal_score,hard_instance_score):
    """

    :param abnormal_score: [T]
    :param hard_instance_score: [800]
    :return:
    """
    max_a, max_a_index = torch.max(abnormal_score, dim=0) # (1)

    # abnormal_size=abnormal_score.shape[0]
    memory_size=hard_instance_score.shape[0]
    #
    # abnormal_score = abnormal_score.squeeze()
    #
    #
    # max_a, max_a_index = torch.max(abnormal_score, dim=1) # (30,1)


    max_a_repeat=max_a.repeat(memory_size) # shape in [memory size]

    assert max_a_repeat.shape[0] ==hard_instance_score.shape[0]

    margin_1=torch.ones_like(max_a_repeat)

    hard_loss=torch.mean(
        F.relu((margin_1 - max_a_repeat + hard_instance_score))
    )

    return hard_loss


def SRF_hard_hinge_loss_remove_one(abnormal_score,hard_instance_score):
    """

    :param abnormal_score: [T]
    :param hard_instance_score: [800]
    :return:
    """
    max_a, max_a_index = torch.max(abnormal_score, dim=0) # (1)

    # abnormal_size=abnormal_score.shape[0]
    memory_size=hard_instance_score.shape[0]
    #
    # abnormal_score = abnormal_score.squeeze()
    #
    #
    # max_a, max_a_index = torch.max(abnormal_score, dim=1) # (30,1)


    max_a_repeat=max_a.repeat(memory_size) # shape in [memory size]

    assert max_a_repeat.shape[0] ==hard_instance_score.shape[0]

    margin_1=torch.ones_like(max_a_repeat)*0.9

    hard_loss=torch.mean(
        F.relu((margin_1-max_a_repeat + hard_instance_score))
    )

    return hard_loss


def SRF_hard_hinge_loss_dynamic_margin(abnormal_score,hard_instance_score,margin_value):
    """

    :param abnormal_score: [T]
    :param hard_instance_score: [800]
    :return:
    """
    max_a, max_a_index = torch.max(abnormal_score, dim=0)  # (1)

    # abnormal_size=abnormal_score.shape[0]
    memory_size=hard_instance_score.shape[0]


    # abnormal_score = abnormal_score.squeeze()
    #
    #
    # max_a, max_a_index = torch.max(abnormal_score, dim=1) # (30,1)
    max_a_repeat=max_a.repeat(memory_size) # shape in [memory size]

    assert max_a_repeat.shape[0] ==hard_instance_score.shape[0]

    margin_1=torch.ones_like(max_a_repeat)*margin_value

    hard_loss=torch.mean(
        F.relu((margin_1-max_a_repeat + hard_instance_score))
    )

    return hard_loss


def SRF_hard_hinge_loss_dynamic_margin_2(abnormal_score,hard_instance_score,margin_value):
    """

    :param abnormal_score: [B,T]
    :param hard_instance_score: [M,1]
    :return:
    """

    max_a, max_a_index = torch.max(abnormal_score, dim=1)  # max_a [B,1]

    # abnormal_size=abnormal_score.shape[0]
    memory_size=hard_instance_score.shape[0]


    max_a_repeat=max_a.unsqueeze(dim=1).repeat(1,memory_size) # shape in [B,memory size]
    hard_instance_score_repeat=hard_instance_score.repeat(1,abnormal_score.shape[0]).permute(1,0)


    assert max_a_repeat.shape[0] ==hard_instance_score_repeat.shape[0]

    margin_1=torch.ones_like(max_a_repeat)*margin_value

    hard_loss=torch.mean(
        F.relu((margin_1-max_a_repeat + hard_instance_score_repeat))
    )

    return hard_loss


def SRF_loss_combine(pred_score,pseudo_y,euc_dis,pred_hard_score,video_label="Abnormal"):
    """
    loss in A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels
    :param pred_score:
    :param pseudo_y:
    :param euc_dis:
    :param pred_hard_score: shape in [800]
    :param video_label: Abnormal or Normal
    :return:
    """
    upper_bound_alpha=torch.tensor([1.0]).cuda()
    hyp=0.05

    # Lr  mse loss in pred_score and
    L_r=L2_loss(pred_score,pseudo_y)

    if video_label in ["Normal"]:
        L_c=euc_dis if euc_dis<upper_bound_alpha else upper_bound_alpha
        hard_hinge_loss=torch.tensor([0.0]).cuda()
    elif video_label in ["Abnormal"]:
        L_c  = 1.0/(euc_dis+1e-8)
        hard_hinge_loss = SRF_hard_hinge_loss_remove_one(pred_score, pred_hard_score)

    else: raise  NotImplementedError(
        "No supported type for videl_label:{}".format(video_label)
    )


    hard_score_loss=torch.mean(pred_hard_score)

    total_loss=L_r+hyp*L_c+hard_hinge_loss+hard_score_loss


    return total_loss,L_r,L_c,hard_hinge_loss,hard_score_loss

def SRF_loss_combine_dynamic_margin(pred_score,pseudo_y,euc_dis,pred_hard_score,video_label="Abnormal",margin_value=1):
    """
    loss in A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels
    FMB loss with dynamic margin
    maring list in [0.6,0.7,0.8,0.9,1.0]
    :param pred_score:
    :param pseudo_y:
    :param euc_dis:
    :param pred_hard_score: shape in [800]
    :param video_label: Abnormal or Normal
    :return:
    """
    upper_bound_alpha=torch.tensor([1.0]).cuda()
    hyp=0.05

    # Lr  mse loss in pred_score and
    L_r=L2_loss(pred_score,pseudo_y)

    if video_label in ["Normal"]:
        L_c=euc_dis if euc_dis<upper_bound_alpha else upper_bound_alpha
        hard_hinge_loss=torch.tensor([0.0]).cuda()
    elif video_label in ["Abnormal"]:
        L_c  = 1.0/(euc_dis+1e-8)
        hard_hinge_loss = SRF_hard_hinge_loss_dynamic_margin(pred_score, pred_hard_score,margin_value)


    else: raise  NotImplementedError(
        "No supported type for videl_label:{}".format(video_label)
    )


    hard_score_loss=torch.mean(pred_hard_score)

    total_loss=L_r+hyp*L_c+hard_hinge_loss+hard_score_loss


    return total_loss,L_r,L_c,hard_hinge_loss,hard_score_loss

def SRF_loss_combine_dynamic_margin_warm_up(pred_score,pseudo_y,euc_dis,pred_hard_score,video_label="Abnormal",margin_value=1,warmup_=True):
    """
    loss in A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels
    FMB loss with dynamic margin
    maring list in [0.6,0.7,0.8,0.9,1.0]
    :param pred_score:
    :param pseudo_y:
    :param euc_dis:
    :param pred_hard_score: shape in [800]
    :param video_label: Abnormal or Normal
    :return:
    """
    upper_bound_alpha=torch.tensor([1.0]).cuda()
    hyp=0.05

    # Lr  mse loss in pred_score and
    L_r=L2_loss(pred_score,pseudo_y)

    if video_label in ["Normal"]:
        L_c=euc_dis if euc_dis<upper_bound_alpha else upper_bound_alpha
        hard_hinge_loss=torch.tensor([0.0]).cuda()
    elif video_label in ["Abnormal"]:
        L_c  = 1.0/(euc_dis+1e-8)
        hard_hinge_loss = SRF_hard_hinge_loss_dynamic_margin(pred_score, pred_hard_score,margin_value)


    else: raise  NotImplementedError(
        "No supported type for videl_label:{}".format(video_label)
    )


    hard_score_loss=torch.mean(pred_hard_score)

    if warmup_:
        total_loss=L_r+hyp*L_c
    else:
        total_loss=L_r+hyp*L_c+hard_hinge_loss+hard_score_loss


    return total_loss,L_r,L_c,hard_hinge_loss,hard_score_loss

def SRF_loss_combine_dynamic_margin_warm_up_2(
        pred_score_abnormal,pseudo_y_abnormal,euc_dis_abnormal,
        pred_score_normal,pseudo_y_normal,euc_dis_normal,
        pred_hard_score,margin_value=1,warmup_=True):
    """
    loss in A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels
    FMB loss with dynamic margin
    maring list in [0.6,0.7,0.8,0.9,1.0]
    pred_score_abnormal in [B,T,1]
    :param pred_score:
    :param pseudo_y:
    :param euc_dis:
    :param pred_hard_score: shape in [800]
    :param video_label: Abnormal or Normal
    :return:
    """

    upper_bound_alpha=torch.tensor([1.0]).cuda()
    hyp=0.05

    # if pred_score_abnormal.ndim==3:
    #     pred_score_abnormal=pred_score_abnormal.squeeze(dim=-1)
    # if pred_score_normal.ndim==3:
    #     pred_score_normal=pred_score_normal.squeeze(dim=-1)

    # Lr  mse loss in pred_score and
    L_r_abnormal=L2_loss(pred_score_abnormal,pseudo_y_abnormal)
    L_r_normal = L2_loss(pred_score_normal, pseudo_y_normal)

    L_r=L_r_abnormal+L_r_normal

    # euc_size=euc_dis_normal.shape[0]
    # for e in range(euc_size):
    #     euc_dis_normal[e]=euc_dis_normal[e] if euc_dis_normal[e] < upper_bound_alpha else upper_bound_alpha
    euc_dis_normal = euc_dis_normal if euc_dis_normal < upper_bound_alpha else upper_bound_alpha

    L_c_normal =torch.mean(euc_dis_normal)

    L_c_abnormal = torch.mean(1.0 / (euc_dis_abnormal + 1e-8))

    L_c=L_c_abnormal+L_c_normal


    hard_hinge_loss = SRF_hard_hinge_loss_dynamic_margin(pred_score_abnormal, pred_hard_score, margin_value)

    # if video_label in ["Normal"]:
    #     L_c=euc_dis if euc_dis<upper_bound_alpha else upper_bound_alpha
    #     hard_hinge_loss=torch.tensor([0.0]).cuda()
    # elif video_label in ["Abnormal"]:
    #     L_c  = 1.0/(euc_dis+1e-8)
    #     hard_hinge_loss = SRF_hard_hinge_loss_dynamic_margin(pred_score, pred_hard_score,margin_value)
    #
    #
    # else: raise  NotImplementedError(
    #     "No supported type for videl_label:{}".format(video_label)
    # )


    hard_score_loss=torch.mean(pred_hard_score)

    if warmup_:
        total_loss=L_r+hyp*L_c
    else:
        total_loss=L_r+hyp*L_c+hard_hinge_loss+hard_score_loss


    return total_loss,L_r,L_c,hard_hinge_loss,hard_score_loss



def SRF_loss_1_dynamic_margin_warm_up(pred_score,pseudo_y,euc_dis,pred_hard_score,video_label="Abnormal",margin_value=1,warmup_=True):
    """
    loss in A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels
    FMB loss with dynamic margin
    maring list in [0.6,0.7,0.8,0.9,1.0]
    :param pred_score:
    :param pseudo_y:
    :param euc_dis:
    :param pred_hard_score: shape in [800]
    :param video_label: Abnormal or Normal
    :return:
    """
    upper_bound_alpha=torch.tensor([1.0]).cuda()
    hyp=0.05

    # Lr  mse loss in pred_score and
    L_r=L2_loss(pred_score,pseudo_y)

    if video_label in ["Normal"]:
        L_c=euc_dis if euc_dis<upper_bound_alpha else upper_bound_alpha
        hard_hinge_loss=torch.tensor([0.0]).cuda()
    elif video_label in ["Abnormal"]:
        L_c  = 1.0/(euc_dis+1e-8)
        hard_hinge_loss = SRF_hard_hinge_loss_dynamic_margin(pred_score, pred_hard_score,margin_value)


    else: raise  NotImplementedError(
        "No supported type for videl_label:{}".format(video_label)
    )


    hard_score_loss=torch.mean(pred_hard_score)

    if warmup_:
        total_loss=L_r+hyp*L_c
    else:
        total_loss=L_r+hyp*L_c+hard_hinge_loss #+hard_score_loss


    return total_loss,L_r,L_c,hard_hinge_loss #,hard_score_loss


def SRF_loss_1(pred_score,pseudo_y,euc_dis,pred_hard_score,video_label="Abnormal"):
    """
    loss in A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels
    :param pred_score:
    :param pseudo_y:
    :param euc_dis:
    :param pred_hard_score: shape in [800]
    :param video_label: Abnormal or Normal
    :return:
    """
    upper_bound_alpha=torch.tensor([1.0]).cuda()
    hyp=0.05

    # Lr  mse loss in pred_score and
    L_r=L2_loss(pred_score,pseudo_y)

    if video_label in ["Normal"]:
        L_c=euc_dis if euc_dis<upper_bound_alpha else upper_bound_alpha
        hard_hinge_loss=torch.tensor([0.0]).cuda()
    elif video_label in ["Abnormal"]:
        L_c  = 1.0/(euc_dis+1e-8)
        hard_hinge_loss = SRF_hard_hinge_loss(pred_score, pred_hard_score)

    else: raise NotImplementedError(
        "No supported type for videl_label:{}".format(video_label)
    )


    # hard_score_loss=torch.mean(pred_hard_score)

    total_loss=L_r+hyp*L_c+hard_hinge_loss #+hard_score_loss


    return total_loss,L_r,L_c,hard_hinge_loss


def SRF_loss_2(pred_score,pseudo_y,euc_dis,pred_hard_score,video_label="Abnormal"):
    """
    loss in A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels
    :param pred_score:
    :param pseudo_y:
    :param euc_dis:
    :param pred_hard_score: shape in [800]
    :param video_label: Abnormal or Normal
    :return:
    """
    upper_bound_alpha=torch.tensor([1.0]).cuda()
    hyp=0.05

    # Lr  L2 loss in pred_score and pseudo label

    L_r=L2_loss(pred_score,pseudo_y)

    if video_label in ["Normal"]:
        L_c=euc_dis if euc_dis<upper_bound_alpha else upper_bound_alpha
        # hard_hinge_loss=torch.tensor([0.0]).cuda()
    elif video_label in ["Abnormal"]:
        L_c  = 1.0/(euc_dis+1e-8)
        # hard_hinge_loss = SRF_hard_hinge_loss(pred_score, pred_hard_score)

    else: raise NotImplementedError(
        "No supported type for videl_label:{}".format(video_label)
    )


    hard_score_loss=torch.mean(pred_hard_score)*2

    total_loss=L_r+hyp*L_c+hard_score_loss


    return total_loss,L_r,L_c,hard_score_loss

_LOSSES={

    # "MSE":L2_loss,
    "COMBINE_LOSS":combine_loss,
    "HARD_COMBINE_LOSS":combine_loss_hard_sample,
    "HARD_LOSS_1":combine_loss_1_hard_sample,
    "HARD_LOSS_2":combine_loss_2_hard_sample,
    "SRF_LOSS":SRF_loss,  # plus loss1 loss2  combine
    "SRF_LOSS_1":SRF_loss_1,
    "SRF_LOSS_2":SRF_loss_2,
    "SRF_LOSS_COMBINE":SRF_loss_combine,
    "SRF_LOSS_COMBINE_DYNAMIC_MARGIN":SRF_loss_combine_dynamic_margin,
    "SRF_LOSS_COMBINE_DYNAMIC_MARGIN_WARMUP":SRF_loss_combine_dynamic_margin_warm_up,
    "SRF_LOSS_COMBINE_DYNAMIC_MARGIN_WARMUP_version2":SRF_loss_combine_dynamic_margin_warm_up_2,
    "SRF_LOSS_1_DYNAMIC_MARGIN_WARMUP":SRF_loss_1_dynamic_margin_warm_up,
}



def get_loss_func(loss_name):

    if loss_name not in _LOSSES.keys():
        raise NotImplementedError(
            "loss {} is not in supported".format(loss_name)
        )
    return _LOSSES[loss_name]


if __name__=="__main__":
    print("loss func")
    # batch size in 30
    # feature [batch_size,32,4096]
    # normal and abnormal shape in [30,32,1]
    # pred score  shape in [batch_size,32]
    # memory bank feature in shape[memory_size,]
    # upper_bound_alpha = torch.tensor([3.0])
    # print(upper_bound_alpha)
    # euc_dis_normal = torch.tensor([2.0, 1, 5, 1, 15, 6])
    # euc_size = euc_dis_normal.shape[0]
    # for e in range(euc_size):
    #     euc_dis_normal[e] = euc_dis_normal[e] if euc_dis_normal[e] < upper_bound_alpha else upper_bound_alpha
    #
    #




    pred=torch.rand(size=[30,64,1])
    hard_score=torch.rand(size=[800,1])

    print(hard_score.repeat(1,156).shape)

    loss=SRF_hard_hinge_loss_dynamic_margin_2(pred,hard_score,1)

    print(loss)

































