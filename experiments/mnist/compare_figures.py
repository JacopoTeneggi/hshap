from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import time
import pandas as pd
import matplotlib.patches as patches

plt.rcParams.update({'font.size': 20})

os.environ["CUDA_VISIBLE_DEVICES"]="7"

device = torch.device("cuda:0")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
torch.manual_seed(0)
model = Net()
weight_path = "mnist_cnn.pt"
model.load_state_dict(torch.load(weight_path, map_location=device)) 
model.to(device)
model.eval()
    
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST("/export/gaon1/data/jteneggi/data/mnist/data", train=True, transform=transform)
test_dataset = datasets.MNIST("/export/gaon1/data/jteneggi/data/mnist/data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

exp_mapper = ["hexp/absolute_0", "hexp/relative_70"]
exp_names = [r"$d$H-SHAP", r"$b$H-SHAP"]

true_positive_id = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    digit = output.argmax(dim=1, keepdim=True).item()
    if digit == target:
        if true_positive_id == 10:
            break
        fig = plt.figure(figsize=(42, 4))
        axes = fig.subplots(1, len(exp_mapper) + 1)
        ax = axes[0]
        ax.imshow(data.cpu().squeeze(0).squeeze(0).numpy())
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title("image")
        for j, exp in enumerate(exp_mapper[:2]):
            ax = axes[j + 1]
            explanation = np.load(os.path.join("true_positive_explanations", f"{exp}/{true_positive_id}.npy"))
            _abs = np.abs(explanation.flatten())
            _max = np.percentile(_abs, 99.9)
            ax.imshow(explanation, cmap="bwr", vmax=_max, vmin=-_max)
            # annotate(ax, image_name)
            ax.set_title(exp_names[j])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)    
        plt.savefig(f"true_positive_explanations/figures/{true_positive_id}.jpg")
        print(f"{true_positive_id} saved figure")
        plt.close()
        true_positive_id += 1

        
    