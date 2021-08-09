# Cross Epoch Learning for Weakly Supervised Anomaly Detection in Surveillance Videos
official implement code for our paper:
Cross-Epoch Learning for Weakly Supervised Anomaly Detection in Surveillance Videos



## Tabel of Contents
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Getting Started](#getting-started)


## Installation
**Requirements**

* CUDA 10.1
* Python=3.6
* PyTorch=1.4.0
* torchvision=0.4.2
* fvcore 
* simplejson
* opencv-python

## Data Preparation
### Shanghai Tech
[ShanghaiTech](https://svip-lab.github.io/datasets.html) is a medium-scale anomaly detection dataset, including 437 videos.
The re-split for weakly supervised task is from [Graph convolutional label noise cleaner](https://arxiv.org/abs/1903.07256)

### UCF-Crime
[UCF-Crime](https://webpages.uncc.edu/cchen62/dataset.html)
a large-scale complex dataset for anomaly detection. 
It contains 13 real-world anomalous behaviors, distributed in 1,900 untrimmed videos with a total duration of 128 hours.


## Getting Started
### step 1
Please make the video data to feature data via [C3D](https://github.com/DavideA/c3d-pytorch) or [I3D](https://github.com/piergiaj/pytorch-i3d)

### step 2 
Before train the model, please check the hyper-parameter file in ./net/config/defaults.py and config/xxx.yaml 
run the train.py if the model,dataset and hyper-parameter is all already. 

### stpe 3 
Inference the mode via inference.py and eval_auc_xxx.py 

