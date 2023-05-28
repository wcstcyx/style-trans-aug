# YOLOX
## 1.项目说明
  本项目为YOLOX模型的训练和测试，本项目在Kaggle网站上运行。
## 2.代码说明
### 2.1Kaggle环境准备
```Python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# 输入数据文件在只读“../input/“ 目录

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
### 2.2下载YOLOX模型
```Python
#函数调用
import warnings
warnings.filterwarnings("ignore")

import ast
import os
import json
import pandas as pd
import torch
import importlib
import cv2 

from shutil import copyfile
from tqdm.notebook import tqdm
tqdm.pandas()
from sklearn.model_selection import GroupKFold
from PIL import Image
from string import Template
from IPython.display import display

#模型下载
!git clone https://github.com/Megvii-BaseDetection/YOLOX -q

%cd YOLOX
!pip install -U pip && pip install -r requirements.txt
!pip install -v -e . 

!pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
### 2.3配置模型文件
```Python
config_file_template = '''

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Define yourself dataset path
        self.data_dir = "/kaggle/input/bird-256/an/train"  #该路径为数据集存放处
        self.train_ann = "train.json"                      #此处为训练集的json文件
        self.val_ann = "val.json"                          #此处为验证集的json文件

        self.num_classes = 1

        self.max_epoch = $max_epoch
        self.data_num_workers = 2
        self.eval_interval = 1
        
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.no_aug_epochs = 2
        
        self.input_size = (256, 256)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (256, 256)
'''

PIPELINE_CONFIG_PATH='cots_config.py'

pipeline = Template(config_file_template).substitute(max_epoch = 5)  #最大epoch数

with open(PIPELINE_CONFIG_PATH, 'w') as f:
    f.write(pipeline)
```
```Python
# ./yolox/data/datasets/voc_classes.py

voc_cls = '''
VOC_CLASSES = (
  "bird",      #训练类别的名称
)
'''
with open('./yolox/data/datasets/voc_classes.py', 'w') as f:
    f.write(voc_cls)

# ./yolox/data/datasets/coco_classes.py

coco_cls = '''
COCO_CLASSES = (
  "bird",       #训练类别的名称
)
'''
with open('./yolox/data/datasets/coco_classes.py', 'w') as f:
    f.write(coco_cls)

# check if everything is ok    
!more ./yolox/data/datasets/coco_classes.py
```
### 2.4下载预训练权重
```Python
sh = 'wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth'  #下载模型权重
MODEL_FILE = 'yolox_l.pth'  #使用的模型权重

with open('script.sh', 'w') as file:
  file.write(sh)

!bash script.sh
```
### 2.5模型训练
```Python
!cp ./tools/train.py ./

!python train.py \
    -f cots_config.py \
    -d 1 \
    -b 8 \
    --fp16 \
    -o \
    -c {MODEL_FILE} 
```
### 2.6测试文件配置
```Python
config_file_template = '''

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        
        # Define yourself dataset path
        self.data_dir = "/kaggle/input/cow-liang-an/liang-an/train"  #测试集的路径
        self.val_ann = "val.json"                                    #测试集的json文件

        self.num_classes = 1

        self.max_epoch = $max_epoch
        self.data_num_workers = 2
        self.eval_interval = 1
        
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.no_aug_epochs = 2
        
        self.input_size = (256, 256)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (256, 256)
'''

PIPELINE_CONFIG_PATH='cots_config.py'

pipeline = Template(config_file_template).substitute(max_epoch = 5)

with open(PIPELINE_CONFIG_PATH, 'w') as f:
    f.write(pipeline)
```
### 2.7模型测试
```Python
! python tools/eval.py \
    -f cots_config.py \
    -d 1 \
    -b 8 \
    --fp16 \
    --conf 0.001 \
    --fuse \
    -c /kaggle/working/YOLOX/YOLOX_outputs/cots_config/best_ckpt.pth   #使用训练效果最好的模型权重
```
