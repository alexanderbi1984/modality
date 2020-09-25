# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.GPUS = (0, 1, 2, 4)
_C.WORKERS = 16
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = True
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height
_C.MODEL.OUTPUT_SIZE = [12, 1]

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'BP4D'
_C.DATASET.TRAINSET = ''
_C.DATASET.TESTSET = ''

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.CONTRAST_FACTOR = 4
_C.DATASET.BRIGHT_FACTOR = 4

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [30, 50]
_C.TRAIN.LR = 0.0001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.WD = 0.0
_C.TRAIN.NESTEROV = False

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 60

_C.TRAIN.RESUME = True
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
