"""
cfg load  and parser
"""
import argparse
import os
from net.config.defaults import get_cfg
# import net.utils.checkpoint as cu


def parse_args():

    parser=argparse.ArgumentParser(
        description=" "
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="path to config file (yaml type)",
        default=r"E:\kaggle\Anomaly-Detection\SRF\configs/UCFCrimeSRF_mini_batch.yaml", # ShanghaiTechSRF.yaml UCFCrimeSRF.yaml
        type=str,
    )
    #ShanghaiTechSRF_epoch.yaml
    #ShanghaiTechSRF_mini_batch.yaml
    #UCFCrimeSRF_version2.yaml
    # parser.add_argument(
    #     "opts",
    #     help="see net/config/defaults.py for more detail",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    return parser.parse_args()


def load_config(args):
    """
    give the  arguments from cmd and yaml
    load and initialize the configs
    :param args:
    :return:
    """
    cfg=get_cfg()
    # load from yaml
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    #load from cmd
    # print(args.opts)
    # if args.opts is not None:
    #     cfg.merge_from_file(args.opts)

    # make a checkpoint dir
    # cu.make_checkpoint_dir(cfg.OUTPUT_DIR,cfg)
    return cfg

if __name__=="__main__":
    print("argparse")
    args=parse_args()
    cfg=load_config(args)
    print(os.listdir("../../configs"))