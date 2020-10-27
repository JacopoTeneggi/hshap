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

HOME = "."

IMG_DIR = os.path.join(HOME, "RSNA_example_images/1")
EXPLANATION_DIR = os.path.join(HOME, "HShap_sick_explanations")
FIGURE_DIR = os.path.join(HOME, "HShap_sick_figures")
patients = os.listdir(EXPLANATION_DIR)
PATIENT_L = len(patients)

preprocess = transforms.Compose(
    [transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()]
)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.600, 0.448, 0.450])

model = models.inception_v3(pretrained=True)
model.aux_logits = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
device = torch.device("cpu")

net_path = os.path.join(HOME, "RSNA_InceptionV3.pth")
model.load_state_dict(torch.load(net_path, map_location=device))

reference_set = datasets.ImageFolder(
    os.path.join(HOME, "single_image_references"), preprocess
)
reference_loader = torch.utils.data.DataLoader(
    reference_set, batch_size=4, shuffle=False
)
reference_iter = iter(reference_loader)
references_img, _ = next(reference_iter)

annotations_path = os.path.join(HOME, "stage_2_train_labels.csv")
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


sizes = [20]
thresholds = [0]
references = [0, 1]
SIZE_L = len(sizes)
THRESH_L = len(thresholds)
REF_L = len(references)
EXP_L = SIZE_L * THRESH_L * REF_L
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
    axes = fig.subplots(REF_L, REF_L)
    for ref_id in references:
        ax = axes[1, ref_id]
        imshow(ax, references_img[ref_id])
        for threshold_id, threshold in enumerate(thresholds):
            for size_id, size in enumerate(sizes):
                explanation = np.load(
                    os.path.join(
                        PATIENT_DIR, "%d_%d_%d.npy" % (threshold_id, size, ref_id)
                    )
                )
            abs = np.abs(explanation.flatten())
            max = np.percentile(abs, 99.9)
            ax = axes[0, ref_id]
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
    plt.savefig(os.path.join(FIGURE_DIR, "%s.eps" % patient), format="eps")
    print(
        "%d/%d Saved figure for patient: %s   Prediction: %d"
        % (patient_id, PATIENT_L, patient, prediction)
    )
    plt.close()
