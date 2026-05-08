#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2026/1/6 16:14:42

@author: LiuKuan
@copyright: Apache License, Version 2.0
'''
from laok.ext.ultralytics_ import train_pose
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ===============================================================================
r'''
调用命令:
    python ./bat_pose_train.py data/bat_pose_train
'''
# ===============================================================================

def train(root, epochs = 50, batch = 2):
    width, height = 640, 640
    kpt_shape = [21, 2]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
    flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12,
                11, 14, 13, 16, 15, 17, 20, 19, 18 ]
    kws = {
    "augment":True,
    "degrees":3,
    "copy_paste": 0.2,
    'train_path':'Image',
    'val_path':'Image_val'
    }
    train_pose(root_dir=root,  flip_idx=flip_idx,
                kpt_shape=kpt_shape, imgsz=(height,width), epochs=epochs, batch=batch, **kws)


if __name__ == '__main__':
    # import fire
    # fire.Fire(train)
    train(root="./20250809_143033", )
