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

# DATA_DIR = "/export/gaon1/data/jteneggi/data/synthetic/LOR"
# df_training = pd.read_json("/export/gaon1/data/jteneggi/data/malaria/training.json")
# df_test = pd.read_json("/export/gaon1/data/jteneggi/data/malaria/test_cropped.json")
# frames = [df_training, df_test]
# df_merged = pd.concat(frames, ignore_index=True)
# ADD IMAGE_NAME COLUMN TO DATAFRAME
# image_names = []
# for i, row in df_merged.iterrows():
#     image_names.append(os.path.basename(row["image"]["pathname"]))
# df_merged["image_name"] = image_names


# def annotate(ax, image_name):
#     query = df_merged.loc[df_merged["image_name"] == image_name]
#     for i, row in query.iterrows():
#         cells = row["objects"]
#         for cell in cells:
#             cell_class = cell["category"]
#             if cell_class == "trophozoite":
#                 bbox = cell["bounding_box"]
#                 upper_left_r = bbox["minimum"]["r"]
#                 upper_left_c = bbox["minimum"]["c"]
#                 lower_right_r = bbox["maximum"]["r"]
#                 lower_right_c = bbox["maximum"]["c"]
#                 w = np.abs(lower_right_c - upper_left_c)
#                 h = np.abs(lower_right_r - upper_left_r)
#                 # Create a Rectangle patch
#                 rect = patches.Rectangle(
#                     (upper_left_c, upper_left_r),
#                     w,
#                     h,
#                     linewidth=2,
#                     edgecolor="k",
#                     facecolor="none",
#                 )
#                 # Add the patch to the Axes
#                 ax.add_patch(rect)


thresholds = ["absolute_0", "relative_50", "relative_60", "relative_70", "relative_80", "relative_90"]

true_positives = np.load("true_positives.npy", allow_pickle=True)
for i, image_path in enumerate(true_positives.item()["1"][:35]):
    fig = plt.figure(figsize=(26, 4))
    axes = fig.subplots(1, len(thresholds) + 1)
    image_name = os.path.basename(image_path)
    print(image_name)
    image = Image.open(image_path)
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("image")
    # annotate(ax, image_name)
    for j, threshold in enumerate(thresholds):
        ax = axes[j + 1]
        explanation = np.load(
            os.path.join("true_positive_explanations", "hexp/%s/%s.npy" % (threshold, image_name))
        )
        _abs = np.abs(explanation.flatten())
        _max = max(_abs)
        # _max = np.percentile(_abs, 99.9)
        # print(_max)
        ax.imshow(explanation, cmap="bwr", vmax=_max, vmin=-_max)
        # annotate(ax, image_name)
        ax.set_title(threshold)
    plt.savefig("true_positive_explanations/figures/%s.eps" % image_name)
    print("%d/%d saved figure" % (i + 1, len(true_positives.item()["1"])))
    plt.close()
