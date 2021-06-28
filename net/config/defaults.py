"""
Configs
"""
from fvcore.common.config import CfgNode
# from net.config import custom_config

# config definition
_C=CfgNode()


# train

_C.TRAIN=CfgNode()

_C.TRAIN.ENABLE=True

_C.TRAIN.DATASET=""

_C.TRAIN.BATCH_SIZE=10

# pre-train model path for fine tune
_C.TRAIN.CHECKPOINT_FILE_PATH= ""

_C.TRAIN.CHECKPOINT_PERIOD=10



#test
_C.TEST=CfgNode()

_C.TEST.DATASET=""

_C.TEST.BATCH_SIZE=10

_C.TEST.CHECKPOINT_FILE_PATH=""

_C.TEST.ENABLE= True



_C.TEST.SAVE_NPY_PATH=r""

_C.TEST.SAVE_FIG =""




#Optimizer  and LR
_C.SOLVER= CfgNode()

_C.SOLVER.BASE_LR=0.01

_C.SOLVER.LR_POLICY="steps"

_C.SOLVER.MAX_EPOCH= 200

_C.SOLVER.MOMENTUM=0.9

_C.SOLVER.WEIGHT_DECAY= 1e-4

_C.SOLVER.BIAS_WEIGHT_DECAY= 0

_C.SOLVER.OPTIMIZING_METHOD=""

_C.SOLVER.BETAS = (0.5, 0.999)

_C.SOLVER.GAMMA = 0.1

_C.SOLVER.DAMPEMING = 0.0

_C.SOLVER.NESTEROV = True

_C.SOLVER.LOSS_FUNC=""

_C.SOLVER.LOSS_TYPES=1

_C.SOLVER.STEP_EPOCHS=[15,25]

_C.SOLVER.STEP_ITERATIONS=[4000,8000]

_C.SOLVER.MAX_ITERATION=10000


# Model
_C.MODEL=CfgNode()

_C.MODEL.MODEL_NAME=""

_C.MODEL.DROPOUT_RATE=0.6


# _C.DATA=CfgNode()
#
# _C.DATA.PATH_TO_DATA_DIR=""




# shanghaitech
_C.TECH=CfgNode()

_C.TECH.PATH_TO_DATA_DIR=r"F:/shanghaitech"

# _C.TECH.INPUT_CHANNEL_NUM=3
#
# _C.TECH.TRAIN_CROP_SIZE= 256
#
# _C.TECH.TEST_CROP_SIZE= 256

# _C.TECH.PIXEL_MAT_FOLDER=""

_C.TECH.FRAME_MAT_FOLDER=""

_C.TECH.TEST_MODE=""



_C.UCF_CRIME_FEATURE=CfgNode()

_C.UCF_CRIME_FEATURE.PATH_TO_DATA_DIR=r""

_C.UCF_CRIME_FEATURE.TEST_MODE=""


_C.MEMORY_BANK=CfgNode()

_C.MEMORY_BANK.BANK_SIZE=0

_C.MEMORY_BANK.DATASET=" "

_C.MEMORY_BANK.BATCH_SIZE=0


_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 1

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

_C.DATA_LOADER.DROP_LAST=True


_C.TENSORBOARD=CfgNode()

_C.TENSORBOARD.PATH=""


#rng seed
_C.RNG_SEED=10

_C.LOG_PERIOD=10

# output dir
_C.OUTPUT_DIR=""

_C.TRAIN_LOGFILE_NAME=""

_C.TEST_LOGFILE_NAME=""

_C.AUC_LOGFILE_NAME=""

_C.NUM_GPUS=1



def get_cfg():

    """

    :return: get a cfg copy
    """
    return _C.clone()


if __name__=="__main__":
    print("defaults configs")

    cfg=get_cfg()
    print(cfg.TRAIN.BATCH_SIZE)
