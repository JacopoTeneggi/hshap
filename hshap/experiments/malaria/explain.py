from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import time
import pandas as pd
import matplotlib.patches as patches
import hshap

os.environ["CUDA_VISIBLE_DEVICES"]="5"

device = torch.device("cuda:0")

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
weight_path = "ResNet18"
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.7206, 0.7204, 0.7651], [0.2305, 0.2384, 0.1706])
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "/export/gaon1/data/jteneggi/data/malaria/trophozoite"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=300, shuffle=True, num_workers=0)
_iter = iter(dataloader)
X, _ = next(_iter)
ref = X.detach().mean(0)
ref = ref.to(device)
#ref = torch.zeros((3, 1200, 1600)).to(device)

hexp = hshap.src.Explainer(model, ref)

DATA_DIR = "/export/gaon1/data/jteneggi/data/malaria/"
df_training = pd.read_json("/export/gaon1/data/jteneggi/data/malaria/training.json")
df_test = pd.read_json("/export/gaon1/data/jteneggi/data/malaria/test_cropped.json")
frames = [df_training, df_test]
df_merged = pd.concat(frames, ignore_index=True)
# ADD IMAGE_NAME COLUMN TO DATAFRAME
image_names = []
for i, row in df_merged.iterrows():
    image_names.append(os.path.basename(row["image"]["pathname"]))
df_merged["image_name"] = image_names

true_positives = np.load("true_positives.npz", allow_pickle=True)
# batch = np.random.choice(true_positives.item()["1"], size=4, replace=False)
# batch = true_positives.item()["1"][:4]
# fig = plt.figure(figsize=(16, 16))
# axes = fig.subplots(2, 2)
for i, image in enumerate(true_positives.item()["1"]):
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    # ax = axes[int(i/2), i%2]
    image_name = os.path.basename(image)
    img = Image.open(image)
    _input = transform(img)
    print(_input.shape)
    ax.imshow(img)
    t0 = time.time()
    hexp_saliency, flatnodes = hexp.explain(_input, label=1, threshold=0, minW=20, minH=20)
    tf = time.time()
    hexp_runtime = round(tf - t0, 6)
    print('Execution completed in %.4f s' % (hexp_runtime))
    abs = np.abs(hexp_saliency.flatten())
    max = np.percentile(abs, 99.9)
    ax.imshow(hexp_saliency, cmap="bwr", alpha=0.75, vmax=max, vmin=-max)
    query = df_merged.loc[df_merged["image_name"] == image_name]
    for i, row in query.iterrows():
        cells = row["objects"]
        for cell in cells:
            cell_class = cell["category"]
            if cell_class == "trophozoite":
                bbox = cell["bounding_box"]
                upper_left_r = bbox["minimum"]["r"]
                upper_left_c = bbox["minimum"]["c"]
                lower_right_r = bbox["maximum"]["r"]
                lower_right_c = bbox["maximum"]["c"]
                w = np.abs(lower_right_c - upper_left_c)
                h = np.abs(lower_right_r - upper_left_r)
                # Create a Rectangle patch
                rect = patches.Rectangle((upper_left_c,upper_left_r),w,h,linewidth=1,edgecolor='r',facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
    ax.set_title(image_name)
    plt.savefig("true_positive_explanations/%s.eps" % image_name)