import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
import os
import glob
import cv2
import numpy as np
from net.dataset.build import DATASET_REGISTRY
from net.utils.parser import parse_args,load_config

@DATASET_REGISTRY.register()
class Shanghaitech(Dataset):
    """
    shanghai tech
    training
        -frames
            -01
                -0000.jpg
    testing
        -frames
            -01
                -000.jpg
    """
    def __init__(self,cfg,mode):
        assert  mode in ["train","training","test","testing"]
        self.data_root=cfg.TECH.PATH_TO_DATA_DIR

        self.mode=mode+"ing"
        self.temporal_length=cfg.TEMPORAL_LENGTH

        self._consturct()
    def _consturct(self):
        """
        recode img path
        last : 12_0175
        :return:
        """
        self.img_paths=[]
        for num_folder in os.listdir(os.path.join(self.data_root, self.mode, "frames")):
        # num_folder="12_0175"
            folder_img = sorted(glob.glob(
                os.path.join(self.data_root, self.mode, "frames", num_folder, "*.jpg").replace("\\", "/")
            ))
            self.img_paths += folder_img[self.temporal_length // 2:(-self.temporal_length // 2)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        past_imgs, center_img, future_imgs, video_idx, img_num = self.load_img_to_rgb(self.img_paths[index])

        past_imgs = self.numpy2tensor(past_imgs)
        center_img = self.numpy2tensor(center_img)
        future_imgs = self.numpy2tensor(future_imgs)


        if self.mode in ["train", "training"]:
            return past_imgs,center_img,future_imgs
        elif self.mode in ["test", "testing"]:
            return past_imgs,center_img,future_imgs, video_idx
        else:
            raise NotImplementedError(
                "mode {} is not supported".format(self.mode)
            )

    def load_img_to_gray(self, path):
        # resize h,w 192,128 256,256

        img_num = int(path.split("\\")[-1].split(".")[0])
        video_idx = (path.split("\\")[0].split("/")[-1])
        for step, i in enumerate(range(-(self.temporal_length // 2), self.temporal_length // 2)):
            img_num_i = img_num + i
            str_img_num_i = "%03d" % img_num_i  # len 3 for each frame
            path_i = path.split("\\")[0] + "/" + str_img_num_i + ".jpg"

            img = cv2.imread(path_i)
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255.0
            if step == 0:
                imgs = np.expand_dims(img, axis=0)
            else:
                imgs = np.concatenate((imgs, np.expand_dims(img, axis=0)), axis=0)

        return imgs, video_idx

    def load_img_to_rgb(self, path):
        # imgs in [h,w,9*3]
        # resize h,w 192,128
        # print("path", path)
        #stack in channel
        img_num = int(path.split("\\")[-1].split(".")[0])
        video_idx = (path.split("\\")[0].split("/")[-1])

        for step, i in enumerate(range(-(self.temporal_length // 2), 0)): # -4,-3,-2,-1
            img_num_i = img_num + i
            img=self.load_single_img_to_rgb(path,img_num_i)
            if step == 0:
                past_imgs = img
            else:
                past_imgs = np.concatenate((past_imgs, img), axis=2)


        center_img=self.load_single_img_to_rgb(path,img_num)


        for step, i in enumerate(range(self.temporal_length // 2 ,0,-1)): # 4,3,2,1

            img_num_i = img_num + i
            img = self.load_single_img_to_rgb(path, img_num_i)
            if step == 0:
                future_imgs = img
            else:
                future_imgs = np.concatenate((future_imgs, img), axis=2)

        return past_imgs,center_img,future_imgs,video_idx,img_num

    def load_single_img_to_rgb(self, path,img_num_i):

        str_img_num_i = "%03d" % img_num_i  # shanghai tech 4 for train 3 for test
        path_i = path.split("\\")[0] + "/" + str_img_num_i + ".jpg"
        img = cv2.imread(path_i)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 127.5) - 1
        return img

    def numpy2tensor(self,np_array):
        np_tensor=torch.from_numpy(np_array)
        np_tensor=np_tensor.permute(2,0,1)
        return  np_tensor




if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args)
    # print(type(cfg))
    data_loader=DataLoader(Shanghaitech(cfg,mode="train"),batch_size=1,shuffle=False)
    print(len(data_loader))
    for step ,(past_imgs,center_img,future_imgs, ) in enumerate(data_loader):
        print("step",step)
        print(past_imgs.shape,center_img.shape,future_imgs.shape)
        # print("video idx and img_num",video_idx,img_num)
