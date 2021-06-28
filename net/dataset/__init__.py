"""
dataset init
for GAN train  image in [-1,1]  /127.5-1
# dataset
UCSDped2 = {'name': 'UCSDped2',
            'path': './UCSDped2',
            'n_clip_train': 16,
            'n_clip_test': 12,
            'ground_truth': [[61, 180], [95, 180], [1, 146], [31, 180], [1, 129], [1, 159],
                             [46, 180], [1, 180], [1, 120], [1, 150], [1, 180], [88, 180]],
            'ground_truth_mask': np.arange(12)+1}

Avenue = {'name': 'Avenue',
          'path': './Avenue',
          'test_mask_path': '../dataset/Avenue/ground_truth_demo/testing_label_mask',
          'n_clip_train': 16,
          'n_clip_test': 21,
          'ground_truth': None,
          'ground_truth_mask': np.arange(21)+1}

"""

from .build import DATASET_REGISTRY # noqa
# from .Avenue import Avenue # noqa
# from .UCSDPed2 import Ucsdped2 # noqa
# from .UCSDPed1 import Ucsdped1 # noqa
# from .ShanghaiTech import Shanghaitech # noqa
# from .Ucf_Crime_Feature import Ucf_Crime_Feature
# from .SVM import one_vs_rest_svm,infer_svm
# from .ShanghaiTech_Feature import Shanghaitech
from  .Ucf_Crime_Feature_SRF import Ucf_Crime_Feature_SRF
# from .XD_Volience_Feature import XD_Violence
from .ShanghaiTech_SRF import SH_SRF
from .ShanghaiTech_SRF_Binary import SH_SRF_Binary
from .ShanghaiTech_SRF_Normal import SH_SRF_Normal
from .Ucf_Crime_Feature_SRF_Binary import Ucf_Crime_Feature_SRF_Binary
from .Ucf_Crime_Feature_SRF_Normal import Ucf_Crime_Feature_SRF_Normal

# from .Ucf_Crime_Feature_SRF_Multi_batch import Ucf_Crime_Feature_SRF_Multi_Batch