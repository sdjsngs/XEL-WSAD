"""
ucf crime class
['Normal','Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
'Shoplifting', 'Stealing', 'Vandalism']

two branch ['Normal' ,'Abnormal' ]
"""
import torch
import  torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
import os
import glob
import cv2
import numpy as np
from PIL import Image
# from net.dataset.build import DATASET_REGISTRY
# from net.utils.parser import parse_args,load_config

# @DATASET_REGISTRY.register()
class Ucf_Crime(Dataset):
    """
    UCF Crime dataset
    trianing:
    800 normal
    810 abnormal
    split it to over-lapped segments  default for 32

    """
    def __init__(self,mode,cfg,):
        assert mode in ["train","test"]
        self.video_root=r"F:\AnomalyDataset\Ucf_Crime_Split" #cfg.AVENUE.PATH_TO_IMG_DIR

        self.mode=mode
        self.temporal_length=16

        self.two_class=['Normal' ,'Abnormal' ]


        self._consturct()

    def _consturct(self):
        """
        Training-Normal-Videos-segment
        Training-Abnormal-Videos-segment

        all_video
        [normal/abnormal,class_name(normal or abnormal class name),video_path]
        {normal:xxxx,
        abnormal: class_name,}
        :return:
        """
        self.normal_video_paths=[]
        self.abnormal_video_paths=[]
        self.total_video_paths=[]


        for num_folder in os.listdir(os.path.join(self.video_root,self.mode,"Normal-Segment")):

            normal_num_folder=os.path.join(
                self.video_root, self.mode, "Normal-Segment",num_folder
            )
            for each_video in os.listdir(normal_num_folder):
                normal_video_pattern=[
                    "Normal","Normal",os.path.join(normal_num_folder,each_video)
                ]

                self.normal_video_paths.append(normal_video_pattern)

        for num_classes in os.listdir(os.path.join(self.video_root, self.mode, "Abnormal-Segment")):

            classes_folder=os.path.join(
                self.video_root, self.mode, "Abnormal-Segment",num_classes
            )
            for num_folder in os.listdir(classes_folder):
                abnormal_num_folder = os.path.join(
                    self.video_root, self.mode, "Abnormal-Segment", num_classes,num_folder
                )
                for each_video in os.listdir(abnormal_num_folder):
                    abnormal_video_pattern = [
                        "Abnormal", num_classes , os.path.join(normal_num_folder, each_video)
                    ]

                    self.abnormal_video_paths.append(abnormal_video_pattern)

        print()

        self.total_video_paths +=self.normal_video_paths
        self.total_video_paths+=self.abnormal_video_paths
        #assert len(self.normal_video_paths) == len(self.abnormal_video_paths), "miss match  in image and flow length "


    def __getitem__(self, index):

        # is normal :[normal ,abnormal]
        # abnormal_class :[13+1]

        is_normal,abnormal_class,seg_path=self.total_video_paths[index]

        seg_tensor=self.load_one_seg(seg_path)


        return



    def load_one_seg(self,path):
        """
        load one segment
        :param path: path to segment
        :return:
        """
        # use opencv-python
        video_cap = cv2.VideoCapture(path)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        frame_buffer = np.zeros((self.temporal_length, frame_height, frame_width, 3), np.dtype("float32"))

        # zero padding if frame_count<self.temporal_length



    def numpy2tensor(self,np_array):
        np_tensor=torch.from_numpy(np_array)
        np_tensor=np_tensor.permute(2,0,1)
        return  np_tensor

    def transfrom_label(self,is_normal):
        """
        one hot label
        [1,0] or [0,1]
        :param is_normal:
        :return:
        """

        return

    def label_update(self):
        """
        updata label in train stage
        :return:
        """
        return
    def __len__(self):
        return len(self.normal_video_paths)


if __name__=="__main__":
    # args=parse_args()
    # cfg=load_config(args)
    # # print(type(cfg))
    cfg=None
    data_loader=DataLoader(Ucf_Crime(mode="train",cfg=cfg),batch_size=10,shuffle=False)
    #
    # for step ,(video,video_index) in enumerate(data_loader):
    #     print("step",step)
    #     print(video.shape)

    print("")
