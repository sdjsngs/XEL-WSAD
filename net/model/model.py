"""
model from
A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels

1 video clip each 16 frame
2 c3d feature extractor
3 fc
    3.1 fc 4096->512

Real-world Anomaly Detection in Surveillance Videos
arXiv:1801.04264v3
WEAKLY SUPERVISED VIDEO ANOMALY DETECTION VIA CENTER-GUIDED DISCRIMINATIVE LEARNING

"""

import torch
import torch.nn as nn
from sklearn.cluster import  KMeans
from kmeans_pytorch import kmeans
import numpy as np
import torch.nn.functional as F
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class SRF_FC(nn.Module):
    def __init__(self, cfg):
        super(SRF_FC, self).__init__()
        self.fc1 = nn.Linear(4096, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(512, 32)
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

        # In the original keras code they use "glorot_normal"
        # As I understand, this is the same as xavier normal in Pytorch
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        """

        :param x: shape in [B,32,4096]
        :return:
        """
        x_1 = self.dropout1(self.relu1(self.fc1(x)))
        # do k-means get euc-dis and pseudo label

        # pseudo_y, euc_dis=self.k_means_cluster(x_1)

        x_2 = self.dropout2(self.fc2(x_1))
        x_3 = self.sig(self.fc3(x_2))

        pred_score=x_3.view(-1)

        return x_1,x_2,x_3,pred_score  #perd_score, pseudo_y, euc_dis



@MODEL_REGISTRY.register()
class MIL_FC(nn.Module):
    def __init__(self, cfg):
        super(MIL_FC, self).__init__()
        self.fc1 = nn.Linear(4096, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(512, 32)
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

        # In the original keras code they use "glorot_normal"
        # As I understand, this is the same as xavier normal in Pytorch
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


@MODEL_REGISTRY.register()
class ARNet_FC(torch.nn.Module):
    # 2048 for I3D combine 1024 for rgb or flow
    def __init__(self, cfg):
        super(ARNet_FC, self).__init__()
        self.feature_dim=1024
        self.fc = nn.Linear(self.feature_dim, self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        # self.apply(weights_init)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.classifier.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, inputs, is_training=True):
        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        score_x=self.sigmoid(self.classifier(x))
        return score_x

        # return x, self.sigmoid(self.classifier(x))



if __name__=="__main__":
    print()

    x=torch.rand(size=[100,64,4096]).cuda()

    model=SRF_FC(cfg=None).cuda()



    x_1,x_2,x_3,pred_score =model(x)


    print()



