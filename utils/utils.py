# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    # def cus_filter(v):
    #     if v[1].requires_grad and 'fc' in v[0] or 'layer4' in v[0]:
    #         return True
    #     else:
    #         return False
    filtered_param = [p[1] for p in model.named_parameters() if p[1].requires_grad and 'fc' in p[0] or 'layer4' in p[0]]  #filter(lambda p:p.requires_grad, model.parameters())
    #filtered_param = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filtered_param,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filtered_param,
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filtered_param,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    torch.save(states, os.path.join(output_dir, filename))
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))
    # The os.symlink() returns EEXIST if link_name already exists
    if os.path.lexists(os.path.join(output_dir, 'latest.pth')):
        print('redirecting latest checkpoint...')
        os.remove(os.path.join(output_dir, 'latest.pth'))
    os.symlink(os.path.join(output_dir, filename), os.path.join(output_dir, 'latest.pth'))

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))  #.module


def f1score(true, pred):
    F1s = []
    precisions = []
    recalls = []
    if true.shape != pred.shape:
        print("Two array must have exactly the same dimension!!")
        return []
    for ix in range(true.shape[1]):
        F1s.append(f1_score(true[:, ix], pred[:, ix]))
        precisions.append(precision_score(true[:, ix], pred[:, ix]))
        recalls.append(recall_score(true[:, ix], pred[:, ix]))
    f1 = np.array(F1s, dtype=np.float32)
    precision = np.array(precisions, dtype=np.float32)
    recall = np.array(recalls, dtype=np.float32)
    return f1, np.mean(f1), precision, np.mean(precision), recall, np.mean(recall)
