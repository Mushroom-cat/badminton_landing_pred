#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2026/1/6 16:16:32

@author: LiuKuan
@copyright: Apache License, Version 2.0
'''
from laok.ext.ultralytics_ import train_det
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ===============================================================================
r'''

命令:
    python .\ball_det_train.py  data/ball_det_train
'''
# ===============================================================================

def train(root, epochs = 100, batch = 2):
    width, height = 640, 640
    train_det(root_dir=root,
              train_path = "Image",
              val_path="Image_val",
              imgsz=(height,width),
              epochs=epochs,
              batch=batch)


if __name__ == '__main__':
    train(root="./",)
    # import fire
    # fire.Fire(train)

