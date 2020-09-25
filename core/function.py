# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.utils import f1score

import time
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, (inp, label, meta) in enumerate(train_loader):

        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        # pred = torch.sigmoid(output).round()

        label = label.cuda(non_blocking=True)
        loss = critertion(output, label)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    # nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    # output_size = config.MODEL.OUTPUT_SIZE

    predictions = None
    all_labels = None
    
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (inp, label,  meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            label = label.cuda(non_blocking=True)

            # loss
            loss = criterion(output, label)
            losses.update(loss.item(), inp.size(0))
            # prediction
            pred = torch.sigmoid(output).round()
            pred = pred.cpu().detach().numpy()

            lab_numpy = label.cpu().numpy()
                        
            if i == 0:
                predictions = pred
                all_labels = lab_numpy
            else:
                predictions = np.concatenate((predictions, pred), axis=0)
                all_labels = np.concatenate((all_labels, lab_numpy), axis=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    np.savetxt('./csv/pred_'+str(epoch)+'.csv', predictions)
    np.savetxt('./csv/label_'+str(epoch)+'.csv', all_labels)

    # Accuracy
    acc = np.mean(np.equal(predictions, all_labels).astype(np.float32))

    f1_arr, mean_f1, _, mean_prec, _, mean_recall = f1score(all_labels, predictions)
    formater = ' '.join(['%.4f'] * len(f1_arr))
    au_msg = 'F1 Score: ' + formater % tuple(f1_arr)

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} f1:{:.4f} precision:{:.4f} recall:{:.4f} accuracy:{:.4f}\n{}'.format(
        epoch, batch_time.avg, losses.avg, mean_f1, mean_prec, mean_recall, acc, au_msg)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalars('valid/metrics', {'Accuracy': acc, 'Mean_F1': mean_f1, 'Mean_precision': mean_prec,
                                            'Mean_recall': mean_recall}, global_steps)
        writer.add_scalars('valid/F1', {'AU1': f1_arr[0], 'AU2': f1_arr[1], 'AU4': f1_arr[2], 'AU6': f1_arr[3],
                                        'AU7': f1_arr[4], 'AU10': f1_arr[5], 'AU12': f1_arr[6], 'AU14': f1_arr[7],
                                        'AU15': f1_arr[8], 'AU17': f1_arr[9], 'AU23': f1_arr[10], 'AU24': f1_arr[11]},
                           global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    predictions = np.array([])

    model.eval()

    end = time.time()
    f1_arr = np.array([])
    with torch.no_grad():
        for i, (inp, label, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            pred = torch.sigmoid(output).round()
            pred = pred.cpu().detach().numpy()

            predictions = np.append(predictions, pred)

            f1, f1_mean = f1score(label.numpy(), pred)
            f1_arr = np.append(f1_arr, f1)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    msg = 'Test Results time:{:.4f} f1:{:.4f}'.format(batch_time.avg, np.mean(f1_arr))
    logger.info(msg)

    return predictions



