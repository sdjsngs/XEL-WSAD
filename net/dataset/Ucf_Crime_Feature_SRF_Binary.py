"""
ucf crime class
['Normal','Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
'Shoplifting', 'Stealing', 'Vandalism']

two branch ['Normal' ,'Abnormal' ]
unmerged video feature
for self-reason framwork
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

from .build import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class Ucf_Crime_Feature_SRF_Binary(Dataset):
    """
    UCF Crime dataset
    trianing:
    800 normal
    810 abnormal
    split it to over-lapped seg
    each video has multi seg  seg_num=video_len//16

    """
    def __init__(self,mode,cfg,):
        assert mode in ["train","test"]
        self.video_root=r"E:\datasets\UCF_C3D_Features_Npy"
        #cfg.UCF_CRIME_FEATURE.PATH_TO_DATA_DIR

        self.cfg=cfg
        self.mode=mode
        self.two_class=['Normal' ,'Abnormal' ]
        self.single_feature_len=4096

        self.normal_feature_paths = []
        self.abnormal_feature_paths = []

        self.test_feature_paths=[]

        self._consturct()
        # self._construct_video()

        self.normal_feature_num=len(self.normal_feature_paths)
        self.abnormal_feature_num = len(self.abnormal_feature_paths)
        self.test_feature_num=len(self.test_feature_paths)

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

        self.normal_feature_paths=(
                glob.glob(
                        os.path.join(
                            self.video_root, self.mode, "Normal", "*.npy"
                        )
                )
            )


        self.abnormal_feature_paths=(
                glob.glob(
                        os.path.join(
                            self.video_root, self.mode, "Abnormal", "*.npy"
                        )
                )
            )


        # if self.mode in ["test"]:
        self.test_feature_paths=self.normal_feature_paths+self.abnormal_feature_paths

    def _load_all_npy(self):
        """
        load all npy
        :return:
        """
        self.normal_npy=np.zeros(shape=[self.normal_feature_num,4096],dtype="float32")
        self.abnormal_npy=np.zeros(shape=[self.abnormal_feature_num,4096],dtype="float32")

        for step, npy_path in enumerate(self.normal_feature_paths):
            self.normal_npy[step]=np.load(npy_path)

        for step, npy_path in enumerate(self.abnormal_feature_paths):
            self.abnormal_npy[step]=np.load(npy_path)

    def __getitem__(self, index):

        # load one feature
        # load
        if self.mode in ["train"]:
            abnormal_feature=self.load_feature(self.abnormal_feature_paths[index])

            normal_index=np.random.randint(0,len(self.normal_feature_paths))

            normal_feature=self.load_feature(self.normal_feature_paths[normal_index])

            return abnormal_feature,normal_feature,normal_index

        elif self.mode in ["test"]:

        # if self.mode in ["test"]:

            feature, label, feature_type = self.load_feature_from_disk(self.test_feature_paths[index])
            video_name=self.test_feature_paths[index].split("\\")[-1].split(".")[0]
            return feature,label,feature_type,video_name

        # return feature,label,feature_type



    def get_class_and_video_name(self,path):
        # 13 abnormal_class 1 normal
        # video name
        abnormal_class = path.split("\\")[-3]
        video_name = path.split("\\")[-2]

        return abnormal_class,video_name
    def load_feature(self,feature_path):
        feature=np.load(feature_path) # [size,4096]
        tensor_feature = torch.from_numpy(feature)

        return tensor_feature


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



    def numpy2tensor(self,np_array):
        np_tensor=torch.from_numpy(np_array)
        np_tensor=np_tensor.permute(2,0,1)
        return  np_tensor



    def __len__(self):
        if self.mode in ["train"]:
            return len(self.abnormal_feature_paths)
        elif self.mode in ["test"]:

            return len(self.normal_feature_paths) + len(self.abnormal_feature_paths)
        else:
            raise  NotImplementedError(
                "unsupported mode,check the  mode  "
            )
        # if self.mode in ["train"]:
        #     return len(self.abnormal_feature_paths)
        # elif self.mode in ["test"]:
        #     return len(self.normal_feature_paths)+len(self.abnormal_feature_paths)
        # else:
        #     raise NotImplementedError(
        #         "Not supported mode:{} in this dataset ".format(self.mode)
        #     )



if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args)
    # # print(type(cfg))
    # cfg=None

    data_loader=DataLoader(Ucf_Crime_Feature_SRF_Binary(mode="train",cfg=cfg),batch_size=1,shuffle=False)
    #
    for step ,(a_feature,n_feature,normal_index) in enumerate(data_loader):
        print("step",step)
        print(a_feature.shape,n_feature.shape)
        print(normal_index.item())




    print("")
