import torch
import  torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
import os
import glob
import cv2
import numpy as np
from PIL import Image
# from net.dataset.build import DATASET_REGISTRY
from net.utils.parser import parse_args,load_config

# @DATASET_REGISTRY.register()
class Avenue(Dataset):
    """
    avenue dataset
    There are total 16 training and 21 testing video sequences.
    Each of the sequences is short; about 1 to 2 minutes long.
    The total number of training frames is 15, 328 and testing frame is 15, 324.
    Resolution of each frame is 640 × 360 pixels.
    training
        -frames
            -01
                -0000.jpg
    load image and flow
    """
    def __init__(self,mode,cfg,):
        assert mode in ["train","training","test","testing"]
        self.img_root=cfg.AVENUE.PATH_TO_IMG_DIR
        self.flow_root=cfg.AVENUE.PATH_TO_FLOW_DIR

        #F:\avenue
        self.mode=mode+"ing"
        self.resolution=256
        self._consturct()

    def _consturct(self):
        """
        recode img path
        :return:
        """
        self.img_paths=[]
        self.flow_paths=[]
        for num_folder in os.listdir(os.path.join(self.img_root,self.mode,"frames")):
            folder_img=sorted(glob.glob(
                os.path.join(self.img_root,self.mode,"frames",num_folder,"*.jpg").replace("\\","/")
            ))
            self.img_paths += folder_img[:-1]

        for num_folder in os.listdir(os.path.join(self.flow_root, self.mode, "optical_flow_visualize")):
            folder_flow = sorted(glob.glob(
                os.path.join(self.flow_root, self.mode, "optical_flow_visualize", num_folder, "*.png").replace("\\", "/")
            ))
            self.flow_paths += folder_flow
        assert len(self.img_paths) == len(self.flow_paths), "miss match  in image and flow length "


    def __getitem__(self, index):

        img,video_idx=self.load_img_to_rgb(self.img_paths[index])
        flow=self.load_single_flow(self)

        img = self.numpy2tensor(img)
        flow = self.numpy2tensor(flow)

        return img,flow,video_idx


    def load_img_to_rgb(self, path):
        # imgs in [h,w,9*3]
        # resize h,w 192,128
        # print("path", path)
        video_idx = (path.split("\\")[0].split("/")[-1])

        center_img=self.load_single_img_to_rgb(path)

        return center_img,video_idx

    def load_single_img_to_rgb(self, path):

        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 127.5) - 1
        return img

    def load_single_flow(self, path):

        flow = Image.open(path)
        flow= flow.convert("RGB")
        flow=np.array(flow)
        flow = np.resize(flow, (self.resolution, self.resolution,3))
        flow = (flow / 127.5) - 1

        return flow

    def numpy2tensor(self,np_array):
        np_tensor=torch.from_numpy(np_array)
        np_tensor=np_tensor.permute(2,0,1)
        return  np_tensor

    def __len__(self):
        return len(self.img_paths)


if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args)
    # print(type(cfg))
    data_loader=DataLoader(Avenue(cfg,mode="test"),batch_size=10,shuffle=False)

    for step ,(video,video_index) in enumerate(data_loader):
        print("step",step)
        print(video.shape)
