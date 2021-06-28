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
class Ucf_Crime_Feature_SRF_Normal(Dataset):
    """
    UCF Crime dataset
    trianing:
    800 normal
    810 abnormal
    split it to over-lapped seg
    each video has multi seg  seg_num=video_len//16

    """
    def __init__(self,mode,cfg):

        assert mode in ["update_epoch"]  # only train data is included
        self.video_root=r"E:\datasets\UCF_C3D_Features_Npy"

        #cfg.UCF_CRIME_FEATURE.PATH_TO_DATA_DIR

        self.cfg=cfg
        self.mode="train"
        self.single_feature_len=4096

        self.normal_feature_paths = []

        self._consturct()

        self.normal_feature_num=len(self.normal_feature_paths)

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


    def __getitem__(self, index):
        """

        :param index:
        :return: [1,T,4096]
        """

        # load normal feature for update memory bank

        normal_feature = self.load_feature(self.normal_feature_paths[index])

        return normal_feature


    def load_feature(self,feature_path):
        feature=np.load(feature_path) # [size,4096]
        tensor_feature = torch.from_numpy(feature)

        return tensor_feature






    def __len__(self):

        return len(self.normal_feature_paths)




if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args)
    # # print(type(cfg))
    # cfg=None

    data_loader=DataLoader(Ucf_Crime_Feature_SRF_Normal(mode="train",cfg=cfg),batch_size=1,shuffle=False)
    #
    for step ,(n_feature) in enumerate(data_loader):
        print("step",step)
        print(n_feature.shape)

    print("")
