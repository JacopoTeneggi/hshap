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
import shap
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

os.environ["CUDA_VISIBLE_DEVICES"]="7"

device = torch.device("cuda:0")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(0)
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

def init_hexp():
    data_dir = "/export/gaon1/data/jteneggi/data/malaria/trophozoite"
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=0)
    _iter = iter(dataloader)
    X, _ = next(_iter)
    ref = X.detach().mean(0)
    ref = ref.to(device)
    # ref = torch.zeros((3, 1200, 1600)).to(device)
    hexp = hshap.src.Explainer(model, ref)
    return hexp

threshold = 0
def hexp_explain(hexp, image):
    explanation, _ = hexp.explain(image, label=1, threshold_mode=threshold_mode, percentile=percentile, threshold=threshold, minW=20, minH=20)
    return explanation

def init_gradexp():
    data_dir = "/export/gaon1/data/jteneggi/data/malaria/trophozoite"
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    _iter = iter(dataloader)
    X, _ = next(_iter)
    X = X.to(device)
    gradexp = shap.GradientExplainer(model, X)
    return gradexp

def gradexp_explain(gradexp, image):
    _input = image.view(-1, 3, 1200, 1600).to(device).detach()
    gradexp_shapley_values, gradexp_indexes = gradexp.shap_values(_input, ranked_outputs=2, nsamples=10)
    explanation = gradexp_shapley_values[0][0].sum(0)
    return explanation

def init_deepexp():
    data_dir = "/export/gaon1/data/jteneggi/data/malaria/trophozoite"
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)
    _iter = iter(dataloader)
    X, _ = next(_iter)
    X = X.to(device)
    deepexp = shap.DeepExplainer(model, X)
    return deepexp

def deepexp_explain(deepexp, image):
    _input = image.view(-1, 3, 1200, 1600).to(device).detach()
    deepexp_shapley_values, deepexp_indexes = deepexp.shap_values(_input, ranked_outputs=2)
    explanation = deepexp_shapley_values[0][0].sum(0)
    return explanation

from shap.maskers import Image as ImageMasker

class TensorImageMasker(ImageMasker):
    def __init__(self, mask, shape=None):
        super().__init__(mask, shape)
    
    def __call__(self, mask, x):
        x = x.cpu().numpy()
        # print("masker", x.shape)
        masked_x = super().__call__(mask, x)
        return masked_x

def init_partexp():
    masker = TensorImageMasker("inpaint_telea", (1200, 1600, 3))   
    def f(x):
        tmp = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
        # print("f", tmp.shape)
        with torch.no_grad():
            output = model(tmp).cpu()
        return output
    partexp = shap.PartitionExplainer(f, masker)
    return partexp

def partexp_explain(partexp, image):
    _input = image.view(-1, 3, 1200, 1600).to(device).permute(0, 2, 3, 1).detach()
    partexp_explanation = partexp(_input, max_evals=300, batch_size=40, outputs=shap.Explanation.argsort.flip[:1])
    partexp_shapley_values = [partexp_explanation.values[..., i] for i in range(partexp_explanation.values.shape[-1])]
    explanation = partexp_shapley_values[0][0].sum(-1)
    return explanation

def gradcam_explain(gradcam, image):
    _input = image.view(-1, 3, 1200, 1600).to(device).detach()
    mask, _ = gradcam(_input)
    explanation = mask.to("cpu").detach().squeeze().numpy()
    return explanation

def gradcampp_explain(gradcampp, image):
    _input = image.view(-1, 3, 1200, 1600).to(device).detach()
    mask, _ = gradcampp(_input)
    explanation = mask.to("cpu").detach().squeeze().numpy()
    return explanation

exp_mapper = [
    {
        "name": "hexp",
        "init": init_hexp,
        "explain": hexp_explain
    },
    {
        "name": "gradexp",
        "init": init_gradexp,
        "explain": gradexp_explain
    },
    {
        "name": "deepexp",
        "init": init_deepexp,
        "explain": deepexp_explain
    },
    {
        "name": "partexp",
        "init": init_partexp,
        "explain": partexp_explain
    },
    {
        "name": "gradcam",
        "init": lambda: GradCAM(model, model.layer4),
        "explain": gradcam_explain
    },
    {
        "name": "gradcampp",
        "init": lambda: GradCAMpp(model, model.layer4),
        "explain": gradcampp_explain
    }
]

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
for exp in [exp_mapper[0]]:
    exp_name = exp["name"]
    explainer = exp["init"]()
    explain = exp["explain"]
    print("Initialized %s" % exp_name)
    for i, image_path in enumerate(true_positives.item()["1"]):
        image_name = os.path.basename(image_path)
        image = transform(Image.open(image_path))
        if exp_name == "hexp":
            for percentile in [50, 60, 70, 80, 90]:
                threshold_mode = "relative"
                t0 = time.time()
                explanation = explain(explainer, image)
                torch.cuda.empty_cache()
                tf = time.time()
                runtime = round(tf - t0, 6)
                print('%s(%d): %d/%d runtime=%.4fs' % (exp_name, percentile, i+1, len(true_positives.item()["1"]), runtime))
                np.save("true_positive_explanations/%s/%s_%d/%s" % (exp_name, threshold_mode, threshold if threshold_mode == "absolute" else percentile, image_name), explanation)
        else:
            t0 = time.time()
            explanation = explain(explainer, image)
            torch.cuda.empty_cache()
            tf = time.time()
            runtime = round(tf - t0, 6)
            print('%s: %d/%d runtime=%.4fs' % (exp_name, i+1, len(true_positives.item()["1"]), runtime))
            np.save("true_positive_explanations/%s/%s" % (exp_name, image_name), explanation)
            