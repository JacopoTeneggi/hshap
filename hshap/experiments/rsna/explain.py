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
import cv2
import HShap as HShap
from PIL import Image

HOME = "."

model = models.inception_v3(pretrained=True)
model.aux_logits = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net_path = os.path.join(HOME, "InceptionV3.pth")
model.load_state_dict(torch.load(net_path, map_location=device))
model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.6, 0.448, 0.450]),
    ]
)

DATA_DIR = os.path.join(HOME, "RSNA_example_images")
EXPLANATION_DIR = os.path.join(HOME, "data/Jacopo/PNA_sick_explanations")
CLASS_DICT = {0: "healthy", 1: "sick"}
testset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), preprocess)
testnames = testset.imgs
batch_size = 4
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4
)
dataiter = iter(testloader)

background = torch.tensor(np.zeros((3, 299, 299), dtype=np.uint8))
hexp = HShap.Explainer(model, background, M=4)
thresholds = [0, 0.5, 1.5]
sizes = [40, 20, 10, 5]

for batch_id, (images, labels) in enumerate(dataiter):
    images = images.to(device)
    labels = labels.to(device)
    for img_id, (image, label) in enumerate(zip(images, labels)):
        ex_id = batch_id * batch_size + img_id
        img_name = testnames[ex_id][0]
        patient_id = os.path.basename(img_name).replace(".png", "")
        print("Example #%d -- patient id is %s" % (ex_id, patient_id))
        if label.item() == 0:
            print("Example image is healthy, skipping for now...")
        else:
            PATIENT_DIR = os.path.join(EXPLANATION_DIR, patient_id)
            if not os.path.exists(PATIENT_DIR):
                os.mkdir(PATIENT_DIR)
            for threshold_id, threshold in enumerate(thresholds):
                for size in sizes:
                    hexp_saliency, _ = hexp.explain(
                        image, label=1, threshold=threshold, minW=size, minH=size
                    )
                    OUT_PATH = os.path.join(
                        PATIENT_DIR, "%d_%d.npy" % (threshold_id, size)
                    )
                    np.save(OUT_PATH, hexp_saliency)
                    print(
                        "Explained patient %s threshold=%.2f size=%d"
                        % (patient_id, threshold, size)
                    )
