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

exp_mapper = ["hexp/absolute_0", "hexp/relative_90", "RDE", "lime"]

true_positives = np.load("true_positives.npy", allow_pickle=True)
for i, image_path in enumerate(true_positives.item()["1"]):
    fig = plt.figure(figsize=(26, 4))
    axes = fig.subplots(1, len(exp_mapper) + 1)
    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("image")
    # annotate(ax, image_name)
    for j, exp in enumerate(exp_mapper):
        ax = axes[j + 1]
        explanation = np.load(os.path.join("true_positive_explanations", "%s/%s.npy" % (exp, image_name)))
        _abs = np.abs(explanation.flatten())
        _max = np.percentile(_abs, 99.9)
        ax.imshow(explanation, cmap="bwr", vmax=_max, vmin=-_max)
        # annotate(ax, image_name)
        ax.set_title(exp)
    plt.savefig("true_positive_explanations/figures/%s.eps" % image_name)
    print("%d/%d saved figure" % (i + 1, len(true_positives.item()["1"])))
    plt.close()
