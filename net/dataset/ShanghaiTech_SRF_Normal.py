"""
shanghai tech

two branch ['Normal' ,'Abnormal' ]
unmerged video feature
for self-reason framwork

    437 videos in resolution 480x856
    data augment anomalousX3
    train 175 normal + 63 anomalous
    train 175 normal + 189 anomalous  anomalous video to three

    test 155 normal + 44 anomalous

normal version only load normal feature

"""
import torch
import  torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
import os
import glob
import cv2
import numpy as np
from PIL import Image


import joblib
from net.utils.parser import parse_args,load_config
#

from .build import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class SH_SRF_Normal(Dataset):
    """
    437 videos in resolution 480x856

    train 175 normal + 63 anomalous
    train 175 normal + 189 anomalous in MIL

    test 155 normal + 44 anomalous

    """
    def __init__(self,mode,cfg,):
        assert mode in ["update_epoch"]
        self.video_root=r"E:\datasets\shanghaitech_C3D_Feature\split_npy"
        #cfg.UCF_CRIME_FEATURE.PATH_TO_DATA_DIR

        self.cfg=cfg
        self.mode="train"
        self.single_feature_len=4096

        self.normal_feature_paths = []
        self.abnormal_feature_paths = []

        self.test_feature_paths=[]

        self._consturct()
        # self._construct_video()

        self.normal_feature_num=len(self.normal_feature_paths)
        # self.abnormal_feature_num = len(self.abnormal_feature_paths)
        # self.test_feature_num=len(self.test_feature_paths)

        # load  features
        # self._load_all_npy()

        # print()

    def _consturct(self):
        """
        Training-Normal-Videos-segment
        Training-Abnormal-Videos-segment
        laod feature

        :return:
        """

        self.normal_feature_paths=glob.glob(
                        os.path.join(
                            self.video_root, self.mode, "Normal", "*.npy"
                        )
                )



        # self.abnormal_feature_paths=glob.glob(
        #                 os.path.join(
        #                     self.video_root, self.mode, "Abnormal", "*.npy"
        #                 )
        #         )
        # if self.mode in ["train"]:
        #     self.abnormal_feature_paths=self.abnormal_feature_paths*3
        #
        #
        # # if self.mode in ["train"]:
        # #     self.test_feature_paths = self.normal_feature_paths + self.abnormal_feature_paths*3
        # # # if self.mode in ["test"]:
        # # else:
        # self.test_feature_paths=self.normal_feature_paths+self.abnormal_feature_paths



    def __getitem__(self, index):

        normal_feature = self.load_feature(self.normal_feature_paths[index])

        return normal_feature

    def get_class_and_video_name(self,path):
        #
        # video name
        abnormal_class = path.split("\\")[-3]
        video_name = path.split("\\")[-2]

        return abnormal_class,video_name

    def load_feature_from_disk(self, feature_path):
        # laod feature from hard disk

        label_dict={
            "Normal":0,
            "Abnormal":1
        }

        feature=np.load(feature_path) # [size,4096]

        feature_type=feature_path.split("\\")[-2]

        featuer_label=np.array([label_dict[feature_type]]*feature.shape[0])

        tensor_feature=torch.from_numpy(feature)

        tensor_label=torch.from_numpy(featuer_label)

        return tensor_feature,tensor_label,feature_type

    def load_feature(self, feature_path):
        feature = np.load(feature_path)  # [size,4096]
        tensor_feature = torch.from_numpy(feature)

        return tensor_feature


    def __len__(self):

        return len(self.normal_feature_paths)

if __name__=="__main__":
    # args=parse_args()
    # cfg=load_config(args)
    # # print(type(cfg))
    # cfg=None

    data_loader=DataLoader(SH_SRF_Normal(mode="test",cfg=None),batch_size=1,shuffle=False)
    #
    for step ,(feature,label,feature_type,video_name) in enumerate(data_loader):
        print("step",step)
        print(feature.shape,label.shape)
        print(type(feature_type))
        print(feature_type,video_name)


    print("")
