# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from utils.transforms import fliplr_joints, image_transform, generate_target, transform_pixel


class BP4D(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        #self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.contrast_factor = cfg.DATASET.CONTRAST_FACTOR


        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.data = pd.read_csv(self.csv_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def random_factor(self):
        scale = 1.0 * (random.uniform(1, 1 + self.scale_factor))
        rot = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0
        is_flip = True if random.random() <= 0.5 and self.flip else False
        contrast = round(random.uniform(1,1+self.contrast_factor),1) if random.random() <= 0.6 else 1.0 
        bright = round(random.uniform(1,4),1) if random.random() <= 0.6 else 1.0 

        trans = {
            'scale':scale,
            'flip':is_flip,
            'contrast':contrast,
            'rot':rot,
            'brightness':bright
        }

        return trans

    # def __getitem__(self, idx):
    #     image_name = self.data.iloc[idx, 0]+'.jpg'
    #     image_path = os.path.join(self.data_root, image_name)
        
    #     label = self.data.iloc[idx, 1:13].values
    #     bbox = self.data.iloc[idx, 13:].values
    #     label = label.astype(np.float32)
    #     bbox = bbox.astype(np.int32)
    #     x, y, w, h = bbox

    #     img = Image.open(image_path)

    #     rot = 0
    #     scale = 1.0 
    #     contrast = 0.0

    #     if self.is_train:
    #         scale, is_flip, contrast, rot = self.random_factor()
    #         if is_flip:
    #             img = Image.fromarray(np.fliplr(img))
    #             x = img.size[0]-(x+w)

    #         new_w = scale*w
    #         new_h = scale*h
    #         new_x = x - (new_w-w)/2.0
    #         new_y = y - (new_h-h)/2.0
    #         bbox = [new_x,new_y,new_w,new_h]
        

    #     img = contrast_resize_crop_rotate(img, contrast, self.input_size, bbox, rot)
        
    #     img = np.array(img, dtype=np.float32)
            
    #     img = (img/255.0 - self.mean) / self.std
    #     img = img.transpose([2, 0, 1])
        
    #     label = torch.Tensor(label)

    #     meta = {'index': idx, 'scale': scale}

    #     return img, label, meta

    #-------------------- version 2 --------read crop file---------------------------
    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]+'.jpg'
        image_path = os.path.join(self.data_root, image_name)
        
        label = self.data.iloc[idx, 1:13].values
        label = label.astype(np.float32)
        
        bbox = self.data.iloc[idx,13:17].values
        bbox = bbox.astype(np.int)

        trans = self.random_factor()
        trans['bbox'] = bbox
        trans['size'] = self.input_size
        
        img = Image.open(image_path)

        img = image_transform(img, trans)

        img = np.array(img,dtype=np.float32)
    
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        
        label = torch.Tensor(label)

        meta = {'index': idx, 'scale': 1.0}

        return img, label, meta
    



if __name__ == '__main__':
    pass
