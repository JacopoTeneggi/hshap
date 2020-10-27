from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import copy
import pandas as pd

HOME = "/home-2/jtenegg1@jhu.edu/"

IMG_DIR = os.path.join(HOME, "data/Jacopo/PNA/test/1")
EXPLANATION_DIR = os.path.join(HOME, "data/Jacopo/PNA_sick_explanations")
patients = os.listdir(EXPLANATION_DIR)
PATIENT_L = len(patients)

preprocess = transforms.Compose(
    [transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()]
)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])

model = models.inception_v3(pretrained=True)
model.aux_logits = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
device = torch.device("cpu")

net_path = os.path.join(HOME, "data/Jacopo/HShap/Pretrained_models/PNA/InceptionV3")
model.load_state_dict(torch.load(net_path, map_location=device))

annotations_path = os.path.join(
    HOME, "data/Jacopo/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
)
df = pd.read_csv(annotations_path)
df_id = df[["patientId"]]
df_x = df[["x"]]
df_y = df[["y"]]
df_width = df[["width"]]
df_height = df[["height"]]
id = df_id.values.flatten()
x = df_x.values.flatten()
y = df_y.values.flatten()
width = df_width.values.flatten()
height = df.height.values.flatten()


def imshow(ax, pic):
    npimg = np.array(pic)
    ax.imshow(np.transpose(npimg, (1, 2, 0)))


sizes = [40, 20, 10, 5]
thresholds = [0, 0.5, 1.5]
SIZE_L = len(sizes)
THRESH_L = len(thresholds)
EXP_L = SIZE_L * THRESH_L
INPUT_SIZE = 299
for patient_id, patient in enumerate(patients):
    pic = Image.open(os.path.join(IMG_DIR, "%s.png" % patient)).convert("RGB")
    pic_w, pic_h = pic.size
    input = preprocess(pic)
    model_input = normalize(input)
    model_input = model_input.to(device)
    input_batch = model_input.unsqueeze(0)
    output = model(input_batch)
    _, prediction = torch.max(output, 1)
    PATIENT_DIR = os.path.join(EXPLANATION_DIR, patient)
    fig = plt.figure(figsize=(11, 15))
    plt.suptitle(
        'Explanation maps for class "sick"\npatient ID: %s\nPredicted: %d'
        % (patient, prediction)
    )
    axes = fig.subplots(SIZE_L, THRESH_L)
    for threshold_id, threshold in enumerate(thresholds):
        for size_id, size in enumerate(sizes):
            explanation = np.load(
                os.path.join(PATIENT_DIR, "%d_%d.npy" % (threshold_id, size))
            )
            abs = np.abs(explanation.flatten())
            max = np.percentile(abs, 99.9)
            ax = axes[size_id, threshold_id]
            imshow(ax, input)
            im = ax.imshow(explanation, cmap="bwr", alpha=0.5, vmax=max, vmin=-max)
            # CREATE BOX PATCHES
            annotations = np.where(id == patient)[0]
            for annotation in annotations:
                box_x = x[annotation]
                box_y = y[annotation]
                box_w = width[annotation]
                box_h = height[annotation]
                rescaled_x = int(box_x / pic_w * INPUT_SIZE)
                rescaled_y = int(box_y / pic_h * INPUT_SIZE)
                rescaled_w = int(box_w / pic_w * INPUT_SIZE)
                rescaled_h = int(box_h / pic_h * INPUT_SIZE)
                box = patches.Rectangle(
                    (rescaled_x, rescaled_y),
                    rescaled_w,
                    rescaled_h,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(box)
            ax.axis("off")
            ax.set_title(
                "min-size = (%d, %d)\nthrehsold=%.2f" % (size, size, threshold)
            )
            fig.colorbar(im, ax=ax)
    plt.savefig(os.path.join(EXPLANATION_DIR, "figures/%s.eps" % patient), format="eps")
    print(
        "%d/%d Saved figure for patient: %s   Prediction: %d"
        % (patient_id, PATIENT_L, patient, prediction)
    )
    plt.close()
