# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.ResModel import ResModel
from models import SimpleCNN
from config import config, update_config
from datasets import get_dataset
from core import function
from utils import utils
# from sync_batchnorm import convert_model


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    #specify which gpu to use
    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    # model = torchvision.models.resnet18(pretrained=config.MODEL.PRETRAINED)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(num_ftrs, config.MODEL.OUTPUT_SIZE[0]))

    model = ResModel(config) 

    

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    # loss
    pos_weight = torch.tensor([2.6, 3.4, 3.0, 1.2, 1.1, 1.0, 1.1, 1.2, 3.4, 1.7, 3.6, 3.8], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()  #
    criterion_val = torch.nn.BCEWithLogitsLoss().cuda()

    optimizer = utils.get_optimizer(config, model)
   
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    dataset_type = get_dataset(config)

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)

        # evaluate
        predictions = function.validate(config, val_loader, model,
                                        criterion_val, epoch, writer_dict)

        if epoch % 5 == 0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'checkpoint_{}.pth'.format(epoch)))
        # utils.save_checkpoint(
        #     {"state_dict": model,
        #      "epoch": epoch + 1,
        #      "optimizer": optimizer.state_dict(),
        #      }, predictions, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)
    main()










