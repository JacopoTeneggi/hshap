import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import cv2
import HShap as HShap
from PIL import Image
import re
import os

HOME = "."

model = models.inception_v3(pretrained=True)
model.aux_logits = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net_path = os.path.join(HOME, "RSNA_InceptionV3.pth")
model.load_state_dict(torch.load(net_path, map_location=device))
model.eval()

all_avg_train = np.load("avg_500_healthy.npy")
background = torch.from_numpy(all_avg_train)
input = background.view(-1, 3, 299, 299).detach()
output = model(input)
softmax = nn.Softmax(dim=1)
softmax_output = softmax(output)
print(softmax_output)

random = np.random()
