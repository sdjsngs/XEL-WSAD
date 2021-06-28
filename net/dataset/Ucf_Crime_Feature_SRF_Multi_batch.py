"""
ucf crime class
['Normal','Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
'Shoplifting', 'Stealing', 'Vandalism']

two branch ['Normal' ,'Abnormal' ]
unmerged video feature
for self-reason framwork
load all feature and get the id number

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
# from net.utils.parser import parse_args,load_config

from .build import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class Ucf_Crime_Feature_SRF_Multi_Batch(Dataset):
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
        # self.find_max_len()
        # self.find_min_len()

        self.temporal_length = 64
        # self.cal_total_features()
        if self.mode in ["train"]:
            self._load_all_npy()

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
        # self.normal_npy=np.zeros(shape=[591770,4096],dtype="float32")
        # self.abnormal_npy=np.zeros(shape=[198501,4096],dtype="float32")

        self.normal_index_list=[]
        self.abnormal_index_list = []

        normal_len_count=0
        abnormal_len_count=0

        for step, npy_path in enumerate(self.normal_feature_paths):

            temp_feature=np.load(npy_path)
            temp_feature_len=temp_feature.shape[0]

            # self.normal_npy[normal_len_count:(normal_len_count+temp_feature_len)]=temp_feature
            normal_len_count+=temp_feature_len
            temp_index=[step]*temp_feature_len

            self.normal_index_list+=temp_index

        for step, npy_path in enumerate(self.abnormal_feature_paths):
            temp_feature = np.load(npy_path)

            temp_feature_len = temp_feature.shape[0]

            # self.abnormal_npy[abnormal_len_count:(abnormal_len_count + temp_feature_len)] = temp_feature

            abnormal_len_count += temp_feature_len
            temp_index = [step] * temp_feature_len

            self.abnormal_index_list += temp_index # [0,0,0,0,1,1,1,....,809,809]

        self.normal_index_arange_list=np.arange(0,len(self.normal_index_list),1).tolist()
        self.abnormal_index_arange_list = np.arange(0, len(self.abnormal_index_list), 1).tolist()

        assert  len(self.normal_index_list)==len(self.normal_index_arange_list)




    def find_max_len(self):
        """
        max temporal len in ucf
        :return:
        """
        self.max_normal_len=0  # 61031
        self.max_abnormal_len=0  # 8868
        for step, npy_path in enumerate(self.normal_feature_paths):
            temp_normal_len=np.load(npy_path).shape[0]
            if self.max_normal_len<temp_normal_len:
                self.max_normal_len=temp_normal_len
        for step, npy_path in enumerate(self.abnormal_feature_paths):
            temp_abnormal_len = np.load(npy_path).shape[0]
            if self.max_abnormal_len < temp_abnormal_len:
                self.max_abnormal_len = temp_abnormal_len

    def find_min_len(self):
        """
        max temporal len in ucf
        :return:
        """
        self.min_normal_len=61031  # 12
        self.min_abnormal_len=8868  # 6
        for step, npy_path in enumerate(self.normal_feature_paths):
            temp_normal_len=np.load(npy_path).shape[0]
            if self.min_normal_len>temp_normal_len:
                self.min_normal_len=temp_normal_len
        for step, npy_path in enumerate(self.abnormal_feature_paths):
            temp_abnormal_len = np.load(npy_path).shape[0]
            if self.min_abnormal_len > temp_abnormal_len:
                self.min_abnormal_len = temp_abnormal_len

    def cut_all_npy_to_seg(self):
        """
        cut
        self.normal_npy=np.zeros(shape=[591770,4096],dtype="float32")
        self.abnormal_npy=np.zeros(shape=[198501,4096],dtype="float32")
        after load all npy
        batch size in 64
        :return:
        """


        self.normal_cut_npys=np.zeros(shape=[])



    def cal_total_features(self):
        """
        max temporal len in ucf
        :return:
        """
        self.normal_len=0 # 591770
        self.abnormal_len=0 # 198501
        for step, npy_path in enumerate(self.normal_feature_paths):
            self.normal_len += np.load(npy_path).shape[0]

        for step, npy_path in enumerate(self.abnormal_feature_paths):
            self.abnormal_len += np.load(npy_path).shape[0]


    def __getitem__(self, index):

        # index in range[]

        # load normal and abnormal feature


        abnormal_feature=self.load_feature_from_index(index,feature_type="Abnormal")

        abnormal_feature=torch.from_numpy(abnormal_feature)

        normal_index=np.random.randint(0,len(self.normal_index_arange_list)//self.temporal_length)
        normal_feature = self.load_feature_from_index(index,feature_type="Normal")

        normal_feature=torch.from_numpy(normal_feature)
        # range in self.temporal_length*index ,self.temporal_length*(index+1)

        if self.mode in ["test"]:# load one feature
            feature, label, feature_type = self.load_feature_from_disk(self.test_feature_paths[index])
            video_name=self.test_feature_paths[index].split("\\")[-1].split(".")[0]
            return feature,label,feature_type,video_name

        return abnormal_feature,normal_feature,normal_index



    def load_feature_from_index(self,start_index,feature_type="Normal"):

        if feature_type in ["Normal"]:
            arange_index_list=self.normal_index_arange_list #[0,1,2,3,4,5...]
            index_list=self.normal_index_list # [0,0,0,1,1,1,1,2,2,2,3,4,4,4,4,...]
            feature_path=self.normal_feature_paths

        elif feature_type in ["Abnormal"]:
            arange_index_list =self.abnormal_index_arange_list
            index_list=self.abnormal_index_list
            feature_path=self.abnormal_feature_paths

        #[0,1,2,3,4,5,6....222222]
        temp_arange_list=arange_index_list[start_index*self.temporal_length:(start_index+1)*self.temporal_length]
        #[0,0,0,0,1,1,1,1,,....,799,799]
        temp_index_list=index_list[start_index*self.temporal_length:(start_index+1)*self.temporal_length]

        set_index_list=set(temp_index_list)

        temp_feature=np.zeros(shape=[len(temp_arange_list),4096],dtype="float32")
        count=0
        for single_set_index in set_index_list: # find feature
            # load
            temp_load_feature = np.load(feature_path[single_set_index])
            start_set_index=index_list.index(single_set_index)

            for temp_index in temp_arange_list:
                # print("temp_index",temp_index)
                feature_index=index_list[temp_index] #  in 0-799 for normal or 0-809 for abnormal

                if feature_index==single_set_index:
                    # take this feature
                    temp_feature[count]=temp_load_feature[temp_index-start_set_index]
                    count+=1

                # else:
                #     break
        # print("count",count)
        assert  count==self.temporal_length



        return temp_feature



    def get_class_and_video_name(self,path):
        # 13 abnormal_class 1 normal
        # video name
        abnormal_class = path.split("\\")[-3]
        video_name = path.split("\\")[-2]

        return abnormal_class,video_name

    def load_feature_from_disk(self, featuer_path):
        # laod feature from hard disk

        label_dict={
            "Normal":0,
            "Abnormal":1
        }

        feature=np.load(featuer_path) # [size,4096]

        feature_type=featuer_path.split("\\")[-2]

        featuer_label=np.array([label_dict[feature_type]]*feature.shape[0])

        tensor_feature=torch.from_numpy(feature)

        tensor_label=torch.from_numpy(featuer_label)

        return tensor_feature,tensor_label,feature_type



    def numpy2tensor(self,np_array):
        np_tensor=torch.from_numpy(np_array)
        np_tensor=np_tensor.permute(2,0,1)
        return np_tensor



    def __len__(self):

        return len(self.abnormal_index_arange_list)//self.temporal_length

        # return len(self.normal_feature_paths) + len(self.abnormal_feature_paths)

        # if self.mode in ["train"]:
        #     return len(self.abnormal_feature_paths)
        # elif self.mode in ["test"]:
        #     return len(self.normal_feature_paths)+len(self.abnormal_feature_paths)
        # else:
        #     raise NotImplementedError(
        #         "Not supported mode:{} in this dataset ".format(self.mode)
        #     )



if __name__=="__main__":
    print()


    # args=parse_args()
    # cfg=load_config(args)
    # # print(type(cfg))
    # cfg=None

    data_loader=DataLoader(Ucf_Crime_Feature_SRF_Multi_Batch(mode="train",cfg=None),batch_size=5,shuffle=False)
    #
    for step ,(abnormal_feature,normal_feature,normal_index) in enumerate(data_loader):
        print("step",step)
        print(abnormal_feature.shape,abnormal_feature.shape)
        print(normal_index.numpy(),type(normal_index.numpy()))




    print("")
