import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
import os
import glob
import cv2
import numpy as np

from net.utils.parser import parse_args,load_config
from net.dataset.build import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class Shanghaitech(Dataset):
    """
    437 videos in resolution 480x856
    data augment anomalousX3
    train 175 normal + 63 anomalous
    train 175 normal + 189 anomalous

    test 155 normal + 44 anomalous
    """

    def __init__(self, mode, cfg, ):
        assert mode in ["train", "test"]
        self.video_root =r"E:\datasets\shanghaitech_C3D_Feature\merged_32"
        #cfg.TECH.PATH_TO_DATA_DIR
        #r"E:\datasets\shanghaitech_C3D_Feature\merged_32"

        self.cfg = cfg
        self.mode = mode
        self.data_mode=cfg.TECH.TEST_MODE if self.mode in ["test"] else self.mode
        #"test_unmerged_l2norm"

        self.single_feature_len = 4096

        self.normal_feature_paths = []
        self.abnormal_feature_paths = []
        self.test_feature_paths = []
        self._consturct()
        # self._construct_video()

        self.normal_feature_num = len(self.normal_feature_paths)
        self.abnormal_feature_num = len(self.abnormal_feature_paths)
        self.test_feature_num = len(self.test_feature_paths)

        # load train normal video and  train  one-vs-rest svm
        if self.mode in ["train"]:
            self._load_all_npy()
        # self.train_svm_and_load_model()
        # print()

    def _consturct(self):
        """
        Training-Normal-Videos-segment
        Training-Abnormal-Videos-segment
        laod feature

        :return:
        """
        self.normal_feature_paths = glob.glob(
            os.path.join(
                self.video_root, self.data_mode, "Normal"
                , "*.npy"
            )
        )
        self.abnormal_feature_paths = glob.glob(
            os.path.join(
                self.video_root, self.data_mode, "Abnormal"
                , "*.npy"
            )
        )

        if self.mode in ["train"]:
            self.abnormal_feature_paths=self.abnormal_feature_paths*3


        if self.mode in ["test"]:
            self.test_feature_paths = self.normal_feature_paths + self.abnormal_feature_paths



    def _load_all_npy(self):
        """
        load all npy
        :return:
        """
        self.normal_npy=[]
        self.abnormal_npy = []
        if self.mode in ["train"]:
            for step, npy_path in enumerate(self.normal_feature_paths):
                self.normal_npy.append(np.load(npy_path).tolist())

            for step, npy_path in enumerate(self.abnormal_feature_paths):
                self.abnormal_npy.append(np.load(npy_path).tolist())
            self.normal_npy=np.array(self.normal_npy)
            self.abnormal_npy = np.array(self.abnormal_npy)
        # elif self.mode in ["test"]:
        #     self.test_npy = np.zeros(shape=[self.test_feature_num, 4096], dtype="float32")
        #     for step, npy_path in enumerate(self.test_feature_paths):
        #         self.test_npy[step] = np.load(npy_path)


    def __getitem__(self, index):

        # is normal :[normal ,abnormal]
        # abnormal_class :[13+1]
        if self.mode in ["train"]:

            # a_feature = self.load_one_feature(self.abnormal_feature_paths[index])
            a_feature = torch.from_numpy(self.abnormal_npy[index])
            # a_label=torch.tensor(np.array([1]*32))
            #abnormal_class, abnormal_video_name = self.get_class_and_video_name(self.abnormal_feature_paths[index])

            # take the normal one
            normal_index = np.random.randint(0, self.normal_feature_num)
            # n_feature = self.load_one_feature(self.normal_feature_paths[normal_index])
            n_feature = torch.from_numpy(self.normal_npy[normal_index])
            # n_label=torch.tensor(np.array([0]*32))
            #normal_class, normal_video_name = self.get_class_and_video_name(self.normal_feature_paths[normal_index])

            return a_feature, n_feature

        elif self.mode in ["test"]:
            # load normal and abnormal one by one
            # return abnormal class  video_name
            feature = self.load_one_feature(self.test_feature_paths[index])
            temp_path = self.test_feature_paths[index]
            abnormal_class = temp_path.split("\\")[-2]
            video_name = temp_path.split("\\")[-1].split(".")[0]

            return feature, [abnormal_class, video_name]

    def get_class_and_video_name(self, path):
        # 13 abnormal_class 1 normal
        # video name
        abnormal_class = path.split("\\")[-3]
        video_name = path.split("\\")[-1].split(".")[0]

        return abnormal_class, video_name

    def load_one_feature(self, path):
        """
        load one segment
        :param path: path to segment
        :return:
        """
        np_array = np.load(path)
        # pred label

        tensor = torch.from_numpy(np_array)
        # return tensor ,tensor_label
        return tensor  # ,tensor_label



    def __len__(self):
        if self.mode in ["train"]:
            return len(self.abnormal_feature_paths)
        elif self.mode in ["test"]:
            return len(self.normal_feature_paths) + len(self.abnormal_feature_paths)
        else:
            raise NotImplementedError(
                "Not supported mode:{} in this dataset ".format(self.mode)
            )


if __name__=="__main__":
    # args=parse_args()
    # cfg=load_config(args)
    cfg=None
    data_loader=DataLoader(Shanghaitech(mode="test",cfg=cfg),batch_size=1,shuffle=False)
    # print(len(data_loader))
    for step ,(a_feature, n_feature ) in enumerate(data_loader):
        print("step",step)
        print(a_feature.shape ) #,n_feature.shape
        # print("video idx and img_num",video_idx,img_num)
