from __future__ import print_function, division

import torch
import torch.nn as nn
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


exp_mapper = ["hexp/absolute_0", "hexp/relative_70", "gradexp", "deepexp", "partexp", "gradcam", "lime"]
exp_names = [r"h-Shap ($\tau = 0$)", r"h-Shap ($\tau = 70\%$)", "GradientExp", "DeepExp", "PartitionExp", "Grad-CAM", "LIME"]

true_positives = np.load("true_positives.npy", allow_pickle=True)
for i, image_path in enumerate(true_positives.item()["1"][:1]):
    fig = plt.figure(figsize=(42, 4))
    axes = fig.subplots(1, len(exp_mapper) + 1)
    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    ax = axes[0]
    ax.imshow(image)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title("image")
    # annotate(ax, image_name)
    for j, exp in enumerate(exp_mapper):
        ax = axes[j + 1]
        explanation = np.load(os.path.join("true_positive_random_explanations", "%s/%s.npy" % (exp, image_name)))
        _abs = np.abs(explanation.flatten())
        _max = np.percentile(_abs, 99.9)
        ax.imshow(explanation, cmap="bwr", vmax=_max, vmin=-_max)
        # annotate(ax, image_name)
        ax.set_title(exp_names[j])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)    
        plt.savefig("true_positive_random_explanations/figures/%s.eps" % image_name)
    print("%d/%d saved figure" % (i + 1, len(true_positives.item()["1"])))
    plt.close()
