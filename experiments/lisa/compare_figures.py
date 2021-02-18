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
from tqdm import tqdm

data_dir = "/export/gaon1/data/jteneggi/data/lisa"
annotations_dir = os.path.join(data_dir, "Annotations", "Annotations")

day_test_sequences = []
for sequence_name in ["daySequence1", "daySequence2"]:
    day_test_sequence_df = pd.read_csv(os.path.join(annotations_dir, sequence_name, "frameAnnotationsBOX.csv"), sep=";")
    day_test_sequences.append(day_test_sequence_df)
day_test_df = pd.concat(day_test_sequences, axis=0)
day_test_df.head()


def annotate(ax, image_name):
    filename = os.path.join("dayTest", image_name)
    annotations = day_test_df.loc[day_test_df["Filename"] == filename]
    for j, annotation in annotations.iterrows():
        if "go" in annotation["Annotation tag"]:
            upper_left_c = int(annotation["Upper left corner X"])
            upper_left_r = int(annotation["Upper left corner Y"])
            lower_right_c = int(annotation["Lower right corner X"])
            lower_right_r = int(annotation["Lower right corner Y"])
            w = np.abs(lower_right_c - upper_left_c)
            h = np.abs(lower_right_r - upper_left_r)
            rect = patches.Rectangle((upper_left_c, upper_left_r), w, h, linewidth=.5, edgecolor="g", facecolor="none")
            ax.add_patch(rect)

exp_mapper = ["hexp/absolute_0", "hexp/relative_50", "gradexp", "deepexp", "partexp", "gradcam", "gradcampp", "RDE", "lime", "lime_fast"]
true_positives = np.load("true_positives.npy", allow_pickle=True)
np.random.seed(0)
examples = np.random.choice(true_positives, size=300, replace=False)
for image_path in tqdm(examples[:3]):
    fig = plt.figure(figsize=(26, 4))
    axes = fig.subplots(1, len(exp_mapper) + 1)
    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("image")
    annotate(ax, image_name)
    for j, exp_name in enumerate(exp_mapper):
        ax = axes[j + 1]
        explanation = np.load(
            os.path.join("true_positive_explanations", exp_name, f"{image_name}.npy")
        )
        _abs = np.abs(explanation.flatten())
        _max = max(_abs)
        ax.imshow(explanation, cmap="bwr")
        annotate(ax, image_name)
        ax.set_title(exp_name)
    plt.savefig("true_positive_explanations/figures/%s.eps" % image_name)
    plt.close()
