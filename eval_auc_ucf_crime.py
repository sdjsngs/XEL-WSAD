"""
eval auc curve

"""
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,auc,average_precision_score
from net.utils.parser import load_config,parse_args
import net.utils.logging_tool as logging
from sklearn import metrics
import os
import scipy.io as scio
import math
logger=logging.get_logger(__name__)

def show_line_one_video(y_score):

    x=np.arange(len(y_score))
    plt.plot(x,y_score)
    plt.show()

def show_score_ground_true(y_score,y_label,title_name,norm_mode,cfg):

    plt.cla()
    plt.title(title_name)
    plt.ylim((0, 1))
    x = np.arange(len(y_score))
    plt.plot(x, y_score,"r-",label="pred_score")
    plt.plot(x,y_label,"g-",label="ground_true")
    plt.legend()  # 添加图例
    save_folder = os.path.join(
        cfg.TEST.SAVE_NPY_PATH, "Temporal_plt",norm_mode
    )
    os.makedirs(save_folder,exist_ok=True)

    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    plt.savefig(os.path.join(
        save_folder, title_name + ".png"
    ))

    # plt.show()

def roc_draw(y_pred_score,y_label):
    """
    draw roc
    :param y_pred:
    :param y_score:
    :return:
    """
    fpr, tpr, thresholds =roc_curve(
        y_label, y_pred_score, pos_label=None, sample_weight=None,drop_intermediate=True
    )
    # plt.title("roc curve")
    # plt.plot(fpr, tpr, marker='o')
    # plt.show()


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
    mat_new=r"F:\SPL_Save_Folder\SRF\UCF_Crime\roc_mat/"+mat_name+str(roc_value)[2:6]+".mat"


    scio.savemat(mat_new, {'X': fpr, "Y": tpr, "description ": "UCF Crime ROC Cruve"+mat_name})


    plt.title("roc curve")
    plt.plot(fpr, tpr, )
    plt.show()



def cal_auc(y_pred,y_label,cfg):
    """
    calculate auc
    :param y_pred:
    :param y_label:
    :return:
    """
    assert len(y_pred)==len(y_label)
    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_pred)


    # save fpr, tpr




    # metrics.auc(fpr, tpr)
    #plt x=fpr,y=tpr
    # plt roc curve img


    rec_auc = auc(fpr, tpr)

    save_fpr_tpr(fpr,tpr,cfg.OUTPUT_DIR,rec_auc)
    plt.title("UCF-Crime  SRF ")
    plt.plot(fpr,tpr)
    plt.show()
    # auc=roc_auc_score(y_label,y_pred)
    return rec_auc





def UCF_GROUND_TRUE(anao_txt):
    """
    load
    F:/AnomalyDataset/Ucf_Crime_Split/annotation/Temporal_Anomaly_Annotation_Time.txt
    :param ANAO_TXT:
    :return:
    """

    r_lines=[]
    total_length=0
    with open(anao_txt,"r") as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip()
            total_length+=(int(line.split("  ")[-1])//16*16)
            r_lines.append(line)

    return r_lines


def ucf_label_pred_score(label_line,pred_array):
    """

    pred array is custom  score or feature nums score

    :param label_line:
    :param pred_array:
    :return:
    """
    #Abuse028_x264.mp4  Abuse  165  240  -1  -1  1412
    video_name,abnormal_class, F_L,F_R,S_L,S_R,T_length =label_line.split("  ")

    F_L=int(F_L)
    F_R=int(F_R)
    S_L=int(S_L)
    S_R=int(S_R)
    T_length=int(T_length)

    pred_scores=[]
    # make  score to T_length each feature contain 16 non-overlap frames
    feature_num=(T_length)//16


    for item in pred_array:
        _item=[item]*16
        pred_scores+=_item

    # ground ture

    ground_ture=[0]*feature_num*16

    if F_L!=-1 and F_R!=-1:
        ground_ture[F_L:F_R+1]=[i+1 for i in ground_ture[F_L:F_R+1]]
    if S_L!=-1 and S_R!=-1:
        ground_ture[S_L:S_R + 1] = [i + 1 for i in ground_ture[S_L:S_R + 1]]
    # # cut ground true drop the last 15 frames (at most )
    # ground_ture=ground_ture[:featuer_num*16]

    assert len(pred_scores)==len(ground_ture) ,"miss match in length of pred score and ground true "
    # draw line to visual
    #show_score_ground_true(pred_scores,ground_ture,video_name)
    return pred_scores,ground_ture

def ucf_label_pred_score_unmerged(label_line,pred_array,cfg):
    """

    pred array is custom  score or feature nums score
    slide windows to do pred
    :param label_line:
    :param pred_array:
    :return:
    """
    #Abuse028_x264.mp4  Abuse  165  240  -1  -1  1412
    video_name,abnormal_class, F_L,F_R,S_L,S_R,T_length =label_line.split("  ")

    F_L=int(F_L)
    F_R=int(F_R)
    S_L=int(S_L)
    S_R=int(S_R)
    T_length=int(T_length)

    pred_scores=[]
    # make score to T_length each feature contain 16 non-overlap frames
    feature_num=(T_length)//16

    assert  int(feature_num)==len(pred_array) ,"miss match in feature num"

    for item in pred_array:
        _item=[item]*16
        pred_scores+=_item


    # ground ture

    ground_ture=[0]*T_length

    if F_L!=-1 and F_R!=-1:
        ground_ture[F_L:F_R+1]=[i+1 for i in ground_ture[F_L:F_R+1]]
    if S_L!=-1 and S_R!=-1:
        ground_ture[S_L:S_R + 1] = [i + 1 for i in ground_ture[S_L:S_R + 1]]
    # # cut ground true drop the last 15 frames (at most )
    ground_ture=ground_ture[:feature_num*16]
    # pred score take the last
    # pred_scores+=[0]*int(T_length-feature_num*16)

    assert len(pred_scores)==len(ground_ture) ,"miss match in length of pred score and ground true "
    # draw line to visual
    #show_score_ground_true(pred_scores,ground_ture,video_name.split(".")[0],"norm",cfg)
    return pred_scores,ground_ture


def ucf_label_pred_score_merged(label_line,pred_array,cfg):
    """
    pred array is 32 score or feature nums score
    1 for abnormal and 0 for normal
    :param label_line:
    :param pred_array:
    :return:
    """
    #Abuse028_x264.mp4  Abuse  165  240  -1  -1  1412
    video_name,abnormal_class, F_L,F_R,S_L,S_R,T_length =label_line.split(" ")

    F_L=int(F_L)
    F_R=int(F_R)
    S_L=int(S_L)
    S_R=int(S_R)
    T_length=int(T_length)

    # if math.isnan(min(pred_array)):
    #     raise RuntimeError(
    #         "ERROR : Got NAN losses {}".format(video_name)
    #     )

    pred_scores=[0]*T_length

    # ground ture
    ground_ture=[0]*T_length

    if F_L!=-1:
        ground_ture[F_L:F_R+1]=[i+1 for i in ground_ture[F_L:F_R+1]]
    if S_L!=-1:
        ground_ture[S_L:S_R + 1] = [i + 1 for i in ground_ture[S_L:S_R + 1]]


    segments_len = T_length // 32


    for i in range(32):
        segment_start_frame = int(i * segments_len)
        segment_end_frame = int((i + 1) * segments_len)
        pred_scores[segment_start_frame:segment_end_frame] = [pred_array[i]]*(segment_end_frame-segment_start_frame)
    pred_scores[int(32 * segments_len):] = [pred_array[-1]] * (len(pred_scores[int(32 * segments_len):]))

    assert len(pred_scores)==len(ground_ture) ,"miss match in length of pred score and ground true "
    # draw line to visual
    # show_score_ground_true(pred_scores,ground_ture,video_name)
    return pred_scores,ground_ture

def get_label_and_score(ano_line,save_folder,cfg):
    y_preds=[]
    y_labels=[]
    for line in ano_line:
        video_name, abnormal_class, F_L, F_R, S_L, S_R, T_length = line.split("  ")
        # load npy
        pred_array=np.load(
            os.path.join(
                save_folder,video_name.split(".")[0]+".npy"
            )
        )
        # merge or unmerged
        # print("cfg.UCF_CRIME_FEATURE.TEST_MODE:{}".format(cfg.UCF_CRIME_FEATURE.TEST_MODE))
        y_pred, y_label = ucf_label_pred_score_unmerged(line, pred_array, cfg)

        # if cfg.UCF_CRIME_FEATURE.TEST_MODE in ["test_merged_l2norm"]:
        #     y_pred,y_label=ucf_label_pred_score_merged(line,pred_array,cfg)
        # elif cfg.UCF_CRIME_FEATURE.TEST_MODE in ["test_unmerged_l2norm"]:
        #     y_pred, y_label = ucf_label_pred_score_unmerged(line, pred_array,cfg)
        y_preds+=y_pred
        y_labels+=y_label
    # y_preds=(np.array(y_preds)/max(np.array(y_preds)))
    return y_preds,y_labels

def eval_auc_roc(cfg):
    """
    load y_pred_score  len = list * cfg.TEST.VIDEO_NUM
    load y_label
    :param cfg:
    :return:
    """
    # logging.setup_logging(cfg.OUTPUT_DIR,cfg.AUC_LOGFILE_NAME)
    # load ground true

    ano_line=UCF_GROUND_TRUE(
        r"E:\datasets\UCFCrime/Temporal_Anomaly_Annotation_Time.txt"
    )
    y_pred_score,y_label=get_label_and_score(
        ano_line,os.path.join(cfg.TEST.SAVE_NPY_PATH,"PRED_TEST_SCORE"),cfg
    )

    auc_values=[]
    assert len(y_pred_score)==len(y_label) ,"len{} and len{}not match".format("y_pred_score","y_label")
    # show_score_ground_true(y_pred_score,y_label,"total")

    auc_value = cal_auc(y_pred_score,y_label,cfg)
    # ap_value=cal_AP(y_pred_score,y_label,cfg)
    print("auc_value:",auc_value)
    # print("ap_value:",ap_value)
    # logger.info("test mode in :{}".format(cfg.UCF_CRIME_FEATURE.TEST_MODE))
    # logger.info("total auc value:{}".format(auc_value))

def show_all_npy(save_score_npy_folder):

    # npy root
    npy_list=os.listdir(save_score_npy_folder)

    for n in npy_list:
        demo=np.load(
            os.path.join(
                save_score_npy_folder,n
            )
        )

        print("video name",n)
        print(max(demo)-min(demo))




if __name__=="__main__":
    """
    load pred score 
    score close to 0 mean anomaly 
    load ground true 
    cal auc value 
    draw roc 
    """

    args=parse_args()
    cfg=load_config(args)


    eval_auc_roc(cfg)










