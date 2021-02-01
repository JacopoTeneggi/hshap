# NUMPY IMPORTS
import numpy as np

# PYTORCH IMPORTS
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

# SYS IMPORTS
import sys
import re
import os
import time
from PIL import Image
import hshap
from hshap.utils import Net
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# HSHAP IMPORTS
# import hshap

# SHAP IMPORTS
# import shap

# GRADCAM, GRADCAM++ IMPORTS
# from gradcam.utils import visualize_cam
# from gradcam import GradCAM, GradCAMpp

# READ ARGVS:
# argvs = sys.argv
# HOME = str(argvs[1])
# EXP_SIZE = int(argvs[2])
# REF_SIZE = int(argvs[3])
# MIN_SIZE = int(argvs[4])
# EXPL_ID = int(argvs[5])

# DEFINE DEVICE
_device = "cuda:0"
device = torch.device(_device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()
print("Current device is {}".format(device))

# LOAD PRE-TRAINED INCEPTION-V3 MODEL
torch.manual_seed(0)
model = Net()
weight_path = "model2.pth"
model.load_state_dict(torch.load(weight_path, map_location=device)) 
model.to(device)
model.eval()
print("Loaded pretrained model")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

data_dir = "/export/gaon1/data/jteneggi/data/synthetic/datasets/"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
_iter = iter(dataloader)
X, _ = next(_iter)
ref = X.detach().mean(0).to(device)
ref_output = model(ref.unsqueeze(0))
ref_logits = torch.nn.Softmax(dim=1)(ref_output)
print(ref_logits)
print("Loaded reference")

exp_mapper = ["hexp/absolute_0", "hexp/relative_70", "gradexp", "deepexp", "partexp", "gradcam", "gradcampp", "naive", "RDE", "lime"]

A = 100*120
exp_x = np.linspace(np.log10(1/A), 0, 100)
relative_perturbation_sizes = np.concatenate(([0], np.sort(10 ** (exp_x))))
perturbation_sizes = np.round(A * relative_perturbation_sizes)
perturbation_sizes = np.array(perturbation_sizes, dtype="int")
print(perturbation_sizes)
perturbations_L = len(perturbation_sizes)

c = [1, 6]
true_positives = np.load("true_positives.npy", allow_pickle=True)
for n in c:
    images = true_positives.item()[str(n)]
    L = len(images)
    for exp_name in exp_mapper:
        LOR_df = pd.DataFrame(columns=["n", "exp_name", "perturbation_size", "logit"])
        exp_logits = torch.zeros((L, perturbations_L)).to(device)
        explanation_dir = os.path.join("true_positive_explanations", exp_name)
        for i, image_path in enumerate(images):
            image_name = os.path.basename(image_path)
            image = transform(Image.open(image_path)).to(device).detach()
            if exp_name == "naive":
                explanation = torch.rand(image.size(1), image.size(2), device=torch.device("cpu")) + .5
            else:
                explanation = np.load(os.path.join(explanation_dir, "%s.npy" % image_name))
            tmp = hshap.utils.compute_perturbed_logits(model, ref, image, explanation, perturbation_sizes, normalization="original")
            exp_logits[i, :] = tmp
            for perturbation, logit in zip(perturbation_sizes, tmp.cpu().numpy()):
                if logit != 0:
                    LOR_df = LOR_df.append({"n": n, "exp_name": exp_name, "perturbation_size": perturbation, "logit": logit}, ignore_index=True)
            print("%s: %d/%d computed perturbed logits" % (exp_name, i+1, L))
        np.save(os.path.join("LOR", "%s/results_%d" % (exp_name, n)), exp_logits.cpu().numpy())
        LOR_df.to_csv(os.path.join("LOR", exp_name, f"results_{n}.csv"))