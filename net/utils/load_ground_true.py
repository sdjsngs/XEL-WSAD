"""
load  ground true
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# avenue testing label mask
from scipy.io import loadmat
import os
# root F:\avenue\pixel ground truth\ground_truth_demo\testing_label_mask




def load_single_mat(mat_file_floder,n_clip=1,dataset="Avenue",vis=True):
    """
    :param mat_file:  mat file path
    :return:  anomaly boundary  [num ,2]
    """
    filename = '%s/%d_label.mat' % (mat_file_floder, n_clip)
    data=sio.loadmat(filename)
    volLabel=data["volLabel"]
    n_bin = np.array([np.sum(volLabel[0, i]) for i in range(len(volLabel[0]))])
    abnormal_frames_index = np.where(n_bin > 0)[0]
    ret=get_segments(abnormal_frames_index)
    # if vis:
    #     show_single_ground_true(n_bin.shape[0],ret)
    return ret

def find_boundary(seq):
    tmp = np.insert(seq, 0, -10)
    diff = tmp[1:] - tmp[:-1]
    peaks = np.where(diff != 1)[0]
    #
    ret = np.empty((len(peaks), 2), dtype=int)
    for i in range(len(ret)):
        ret[i] = [peaks[i], (peaks[i+1]-1) if i < len(ret)-1 else (len(seq)-1)]
    return ret

def get_segments(seq):

    #
    ends = find_boundary(seq)
    # segment=np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]).reshape(-1) + 1  # +1 for 1-based index (same as UCSD data)
    segment = np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]) # .reshape(-1)
    return segment


def create_avenue_label(n_bin,abnormal_frames_idnex):
    """
    :param n_bin: video len
    :param abnormal_frames_idnex:
    :return:
    """
    one_label=[1]*len(n_bin)
    for index in abnormal_frames_idnex:
        one_label[index]=0
    # cut in [8:-7]
    return one_label[1:-1]

def load_ground_truth_Avenue(folder, n_clip):
    ret = []
    for i in range(n_clip):
        filename = '%s/%d_label.mat' % (folder, i+1)
        # print(filename)
        data = loadmat(filename)['volLabel']
        n_bin = np.array([np.sum(data[0, i]) for i in range(len(data[0]))])

        abnormal_frames = np.where(n_bin > 0)[0]

        ret.append(create_avenue_label(n_bin,abnormal_frames))
    return ret


def show_single_ground_true(time_druation,anomaly_boundry,cfg=None):
    y=np.zeros(time_druation)
    for boundry in anomaly_boundry:
        y[boundry[0]:boundry[1]]=1
    x=np.arange(time_druation)
    plt.stackplot(x,y,colors='red')
    plt.show()
    return


def create_gt_label(start_index,end_index,length):
    """
    create 0,1 label
    1 for normal
    0 for abnormal
    :param start_index:
    :param end_index:
    :param length:
    :return:
    """
    left=[1]*(start_index-1)
    right=[1]*(length-end_index)
    mid=[0]*(end_index-start_index+1)

    total=left+mid+right
    return total[:-1]

def load_gt_ucsd_ped2(filename):
    """
    load ground true in ucsd ped2 dataset
    ucsd show 12 [] list for anomaly time location
    uscd gt start from 1
    1 for normal
    0 for abnormal
    :param filename: ped2.mat
    :return:
    """
    # video len for uscd ped2 testing video
    video_len=[
        180,180,150,180,150,180,180,180,120,150,180,180
    ]
    data = sio.loadmat(filename)
    gt=data["gt"][0]   # [array(),array(),......]
    uscd_gt=[]
    for video_num in range(len(video_len)):
        gt_one_clip=create_gt_label(gt[video_num][0][0],gt[video_num][1][0],video_len[video_num])
        uscd_gt.append(gt_one_clip)
    return uscd_gt

def reverse_label(single_npy):
    """
    reverse the normal/abnormal label for shanghaitech
    :return:
    """
    reverse_npy=1-single_npy
    reverse_npy=reverse_npy[1:-1]
    return reverse_npy


def load_shanghaitech(folder,n_clip=107,mode="frame"):
    """
    load .npy file for shanghai tech dataset
    it contain pixel-level and frame-level
    1 for abnormal and 0 for normal
    :param folder:
    :param n_clip:
    :param mode:
    :return: 1 for normal and 0 for abnormal
    """
    assert mode in ["frame","pixel"]
    # mode="test_%s_mask" %(mode)
    shanghaitech_gt=[]
    for singel_npy in os.listdir(folder):
        filename ='%s/%s' % (folder, singel_npy)
        npy_label=np.load(filename)
        reverse_npy_label=list(reverse_label(npy_label))
        shanghaitech_gt.append(reverse_npy_label)

    return shanghaitech_gt





if __name__=="__main__":
    print("ground true ")
    # root=r"F:\avenue\pixel ground truth\ground_truth_demo\testing_label_mask/"
    # singel_mat=root+"1_label.mat"
    # vol=load_single_mat(root,8)
    # print(vol)
    # load_ground_truth_Avenue(folder=root,n_clip=1)

    # ret=load_ground_truth_Avenue(root,len(os.listdir(root)))
    # # print(ret.shape)
    # filename=r"D:\AnomalyDataset\ped2/ped2.mat"
    # data = sio.loadmat(filename)

    # print((data["gt"][0]))
    folder=r"F:\shanghaitech\testing\test_frame_mask"
    # pixel=r"F:\shanghaitech\testing\test_pixel_mask/01_0014.npy"
    # load_shanghaitech()
    # a=np.load(folder)
    mask=r"D:\dataset\shanghaitech\testing\test_frame_mask/01_0014.npy"
    # load_shanghaitech(folder)
    # print(len(reverse_label(a)))
    a=np.load(mask)

    print()






