from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class ConvBN(nn.Module):

    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(ConvBN, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class SimpleCNN(nn.Module):
    expansion = 1

    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        fd_base = 16
        self.conv1 = ConvBN(7, 3, fd_base, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBN(3, fd_base, fd_base * 2, 2)
        self.conv3 = ConvBN(3, fd_base * 2, fd_base * 4, 2)
        self.conv4 = ConvBN(3, fd_base * 4, fd_base * 8, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(fd_base * 8, config.MODEL.OUTPUT_SIZE[0])

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x
