"""
logging
"""


import  logging
import sys
import os
import decimal
import simplejson
from fvcore.common.file_io import PathManager


def _cached_log_stream(filename):
    return PathManager.open(filename, "a")

def setup_logging(output_dir=None,log_name=None):


    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)

    _FORMAT=logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m%d %H:%M:%S",
    )
    # logging.basicConfig(
    #     level=logging.INFO, format=_FORMAT, stream=sys.stdout
    # )

    sh=logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(_FORMAT)
    logger.addHandler(sh)

    out_log=os.path.join(output_dir,log_name)
    fh=logging.FileHandler(out_log) #_open_log_file(out_log)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FORMAT)
    logger.addHandler(fh)

def get_logger(name):
    """
    get logging with the name
    :param name: (string)
    :return:
    """
    return logging.getLogger(name)


def log_json_stats(stats):

    stats={
        k:decimal.Decimal("{:.6f}".format(v)) if isinstance(v,float) else v
            for k ,v in stats.items()
    }
    json_stats=simplejson.dumps(stats,sort_keys=False,use_decimal=True)
    logger=get_logger(__name__)
    logger.info("json_stats:{:s}".format(json_stats))




if __name__=="__main__":
    print("logging file ")
    print(__name__)