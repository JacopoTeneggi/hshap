# NUMPY IMPORTS
import numpy as np

# PYTORCH IMPORTS
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

# SYS IMPORTS
import os
import sys
import shutil
from pathlib import Path

# HSHAP IMPORTS
import hshap

# READ ARGVS
argvs = sys.argv
exp_size = int(argvs[1])

# DEFINE GLOBAL CONSTANTS
HOME = "/home/jacopo"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.299, 0.224, 0.225])
DATA_DIR = os.path.join(HOME, "repo/hshap/data/rsna/datasets")
SICK_IMGS_DIR = os.path.join(DATA_DIR, "test/sick")
SAMPLE_DIR = os.path.join(HOME, "repo/hshap/data/rsna/LOR/{}".format(exp_size))

# MAKE OUTPUT DIR
Path(SAMPLE_DIR).mkdir(parents=True, exist_ok=True)

# EXTRACT RANDOM SAMPLE FROM SICK IMAGES
imgs = os.listdir(SICK_IMGS_DIR)
L = len(imgs)
sample = np.random.choice(np.arange(L), size=exp_size, replace=False)

for i, img_id in enumerate(sample):
    img_name = imgs[img_id]
    img_path = os.path.join(SICK_IMGS_DIR, img_name)
    shutil.copy(img_path, SAMPLE_DIR)
    print(img_name, i)
