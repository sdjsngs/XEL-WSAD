"""
sklearn k-means
"""
import os
import numpy as np
from sklearn.cluster import KMeans



def binary_cluster(input_x,center_num=2):
    """"""

    return



if __name__=="__main__":
    print()

    demo_x=np.random.random(size=[1000,4096])


    cls=KMeans(n_clusters=2,random_state=0).fit(demo_x)

    pred_label=cls.labels_

    center_feature=cls.cluster_centers_

    print()