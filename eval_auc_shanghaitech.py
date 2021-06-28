"""
eval auc curve
pred score in npy
ground true in mat
"""
import os
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,auc
from net.utils.parser import load_config,parse_args
import net.utils.logging_tool as logging
from sklearn import metrics
from net.utils.load_ground_true import load_shanghaitech
import scipy.io as scio
logger=logging.get_logger(__name__)


def save_fpr_tpr(fpr, tpr,mat_name,roc_value):
    """
    draw roc
    :param y_pred:
    :param y_score:
    :return:
    """
    fpr=np.expand_dims(fpr,axis=1)
    tpr=np.expand_dims(tpr,axis=1)
    mat_name=mat_name.split("/")[-1]

    mat_new=r"F:\SPL_Save_Folder\SRF\SH\roc_mat/"+mat_name+str(roc_value)[2:6]+".mat"


    scio.savemat(mat_new, {'X': fpr, "Y": tpr, "description ": "SH ROC Cruve"+mat_name})


    # plt.title("roc curve")
    # plt.plot(fpr, tpr,)
    # plt.show()

def remove_edge(plt):
    """
    visual ground in non-line bar
    :param plt:
    :return:
    """
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def show_ground_true(y,score):
    # ax = plt.gca()  # 获取到当前坐标轴信息
    # ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    # ax.invert_yaxis()  # 反转Y坐标轴
    plt.xlim((0, len(y)))
    plt.ylim((0, 1.01))
    x=np.arange(len(y))
    plt.plot(x, score,"r")
    plt.bar(x,y,width=1)

    plt.show()
def show_score_ground_true(y_score,y_label,title_name,norm_mode,cfg):

    plt.cla()
    plt.title(title_name)
    plt.ylim((0, 1))
    x = np.arange(len(y_score))
    plt.plot(x, y_score,"r-",label="pred_score")
    plt.plot(x,y_label,"g-",label="ground_true")
    plt.legend()  # 添加图例
    # save folder
    save_folder=os.path.join(
        cfg.TEST.SAVE_NPY_PATH,norm_mode
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(
        save_folder,title_name+".png"
    ))
    # plt.show()
def show_line_one_video(y_score):

    x=np.arange(len(y_score))
    plt.plot(x,y_score)
    plt.show()

def show_pred_score_and_ground_true(y_score,y_label):
    x = np.arange(len(y_score))
    plt.plot(x, y_score,"r--")
    plt.plot(x,y_label,"g--")
    plt.show()

def roc_draw(y_pred_score,y_label):
    """
    draw roc
    :param y_pred:
    :param y_score:
    :return:
    """
    fpr, tpr, thresholds =roc_curve(y_label, y_pred_score, pos_label=None, sample_weight=None,

                              drop_intermediate=True)

    plt.title("roc curve")
    plt.plot(fpr, tpr, marker='o')
    plt.show()

def cal_auc(y_pred,y_label,cfg):
    """
    calculate auc
    :param y_pred:
    :param y_label:
    :return:
    """
    assert len(y_pred)==len(y_label)
    # auc=roc_auc_score(y_label,y_pred)

    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_pred)


    # metrics.auc(fpr, tpr)
    # plt x=fpr,y=tpr
    rec_auc = auc(fpr, tpr)


    plt.title("shanghai tech ")
    plt.plot(fpr, tpr)
    plt.show()

    save_fpr_tpr(fpr,tpr,cfg.OUTPUT_DIR,rec_auc)

    # auc=roc_auc_score(y_label,y_pred)
    return rec_auc


def SH_GROUND_TRUE_ANNO(anao_txt):
    """
    load
    D:\dataset\ShanghaiTech_new_split/SH_Test_Annotate.txt
    :param ANAO_TXT:
    :return:
    """

    r_lines=[]
    # total_length=0
    with open(anao_txt,"r") as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip()
            # total_length+=(int(line.split("  ")[-1])//16*16)
            r_lines.append(line)

    return r_lines


def sh_label_pred_score_unmerged(label_line,pred_array,cfg):
    """

    pred array is custom  score or feature nums score
    slide windows to do pred
    :param label_line:
    :param pred_array:
    :return:
    """
    #01_001 normal 764
    video_name,abnormal_class, T_length =label_line.split(" ")


    T_length=int(T_length)

    pred_scores=[]
    # make  score to T_length each feature contain 16 non-overlap frames
    feature_num=(T_length)//16


    for item in pred_array:
        _item=[item]*16
        pred_scores+=_item

    # ground ture
    if abnormal_class  in ["Normal"]:
        ground_ture=[0]*feature_num*16
    elif abnormal_class  in ["Abnormal"]:
        ground_ture=load_one_tech_test_npy_anno(video_name).tolist()
        ground_ture=ground_ture[:feature_num*16]


    assert len(pred_scores)==len(ground_ture) ,"miss match in length of pred score and ground true "
    # draw line to visual
    #show_score_ground_true(pred_scores,ground_ture,abnormal_class+"_"+video_name,"all_norm",cfg)
    return pred_scores,ground_ture


def sh_label_pred_score_merged(label_line,pred_array,cfg):
    """
    pred array is 32 score or feature nums score
    1 for abnormal and 0 for normal
    :param label_line:
    :param pred_array:
    :return:
    """
    #Abuse028_x264.mp4  Abuse  165  240  -1  -1  1412
    video_name,abnormal_class,T_length =label_line.split(" ")

    T_length=int(T_length)

    pred_scores=[0]*T_length


    # ground ture
    if abnormal_class in ["Normal"]:
        ground_ture = [0] * T_length
    elif abnormal_class in ["Abnormal"]:
        ground_ture = load_one_tech_test_npy_anno(video_name).tolist()

    segments_len = T_length // 32


    for i in range(32):
        segment_start_frame = int(i * segments_len)
        segment_end_frame = int((i + 1) * segments_len)
        pred_scores[segment_start_frame:segment_end_frame] = [pred_array[i]]*(segment_end_frame-segment_start_frame)
    # pred_scores[int(32 * segments_len):] = [pred_array[-1]] * (len(pred_scores[int(32 * segments_len):]))

    assert len(pred_scores)==len(ground_ture) ,"miss match in length of pred score and ground true "
    # draw line to visual
    #show_score_ground_true(pred_scores,ground_ture,abnormal_class+"_"+video_name,"no_norm",cfg)
    return pred_scores,ground_ture


def load_one_tech_test_npy_anno(video_name):
    # frame mask
    # and pixel mask
    frame_mask_root=r"D:\AnomalyDataset\shanghaitech\testing\test_frame_mask"

    gt=np.load(
        os.path.join(
            frame_mask_root,video_name+".npy"
        )
    )
    return gt


def get_label_and_score(ano_line,save_folder,cfg):
    y_preds = []
    y_labels = []
    for line in ano_line:

        video_name, abnormal_class, T_length = line.split(" ")
        # load npy
        pred_array = np.load(
            os.path.join(
                save_folder,  video_name + ".npy"
            )
        )


        y_pred, y_label = sh_label_pred_score_unmerged(line, pred_array,cfg)

        y_preds += y_pred
        y_labels += y_label

    # y_preds=norm_min_max(np.array(y_preds)).tolist()
    # y_preds=[max(0,x-0.225) for x in y_preds]
    return y_preds, y_labels


def eval_auc_roc(cfg):
    """
    load y_pred_score  len = list * cfg.TEST.VIDEO_NUM
    load y_label {0,1} 0 for abnormal  1 for normal
    :param cfg:
    :return:
    """
    logging.setup_logging(cfg.OUTPUT_DIR,cfg.AUC_LOGFILE_NAME)

    ano_line=SH_GROUND_TRUE_ANNO(
        r"E:\datasets\shanghaitech_C3D_Feature/SH_Test_Annotate.txt"
    )
    y_pred_score,y_label=get_label_and_score(
        ano_line,os.path.join(cfg.TEST.SAVE_NPY_PATH,"PRED_TEST_SCORE"),cfg
    )

    # y_pred_score=load_npy_tech(cfg.TEST.SAVE_NPY_PATH,cfg.TEST.PATH,cfg)
    # y_label=load_shanghaitech(cfg.TECH.FRAME_MAT_FOLDER)
    auc_values=[]
    assert len(y_pred_score)==len(y_label) ,"len{} and len{}not match".format("y_pred_score","y_label")
    # logger.info("auc for each video and all video ")

    auc_value = cal_auc(y_pred_score, y_label,cfg)
    # # roc_draw(total_y_pred, total_y_label)
    # logger.info("total auc value:{}".format(auc_value))
    print("total auc value:{}".format(auc_value))




if __name__=="__main__":
    """
    load pred score 
    score close to 0 mean anomaly 
    load ground true  
    draw roc  
    """
    args=parse_args()
    cfg=load_config(args)

    eval_auc_roc(cfg)





