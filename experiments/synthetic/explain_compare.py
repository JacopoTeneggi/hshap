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
from net import Net
import shap
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from RDE.ComputeExplainability import generate_explainability_map 
from lime import lime_image

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.normal_(m.bias.data)

os.environ["CUDA_VISIBLE_DEVICES"]="7"

device = torch.device("cuda:0")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# torch.manual_seed(0)
model = Net()
# model.apply(weights_init)
weight_path = "model2.pth"
model.load_state_dict(torch.load(weight_path, map_location=device)) 
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def init_hexp():
    data_dir = "/export/gaon1/data/jteneggi/data/synthetic/datasets"
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    _iter = iter(dataloader)
    X, _ = next(_iter)
    ref = X.detach().mean(0)
    ref = ref.to(device)
    hexp = hshap.src.Explainer(model, ref)
    return hexp

def hexp_explain(hexp, image_t):
    (explanation, _) = hexp.explain(image_t, label=1, minW=2, minH=2, threshold_mode=threshold_mode, percentile=percentile, threshold=threshold)
    return explanation

def init_gradexp():
    data_dir = "/export/gaon1/data/jteneggi/data/synthetic/datasets"
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    _iter = iter(dataloader)
    X, _ = next(_iter)
    X = X.to(device)
    gradexp = shap.GradientExplainer(model, X)
    return gradexp

def gradexp_explain(gradexp, image_t):
    _input = image_t.unsqueeze(0).to(device).detach()
    gradexp_shapley_values, gradexp_indexes = gradexp.shap_values(_input, ranked_outputs=2, nsamples=200)
    explanation = gradexp_shapley_values[0][0].sum(0)
    return explanation

def init_deepexp():
    data_dir = "/export/gaon1/data/jteneggi/data/synthetic/datasets"
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    _iter = iter(dataloader)
    X, _ = next(_iter)
    X = X.to(device)
    deepexp = shap.DeepExplainer(model, X)
    return deepexp

def deepexp_explain(deepexp, image_t):
    _input = image_t.unsqueeze(0).to(device).detach()
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
    # data_dir = "/export/gaon1/data/jteneggi/data/synthetic/datasets"
    # train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    # _iter = iter(dataloader)
    # X, _ = next(_iter)
    # ref = X.detach().mean(0).numpy()
    masker = TensorImageMasker("inpaint_telea", (100, 120, 3))   
    def f(x):
        tmp = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
        # print("f", tmp.shape)
        with torch.no_grad():
            output = model(tmp).cpu()
        return output
    partexp = shap.PartitionExplainer(f, masker)
    return partexp

def partexp_explain(partexp, image_t):
    _input = image_t.unsqueeze(0).to(device).permute(0, 2, 3, 1).detach()
    partexp_explanation = partexp(_input, max_evals=500, batch_size=100, outputs=shap.Explanation.argsort.flip[:1])
    partexp_shapley_values = [partexp_explanation.values[..., i] for i in range(partexp_explanation.values.shape[-1])]
    explanation = partexp_shapley_values[0][0].sum(-1)
    return explanation

def gradcam_explain(gradcam, image_t):
    _input = image_t.unsqueeze(0).to(device).detach()
    mask, _ = gradcam(_input)
    explanation = mask.to("cpu").detach().squeeze().numpy()
    return explanation

def gradcampp_explain(gradcampp, image_t):
    _input = image_t.unsqueeze(0).to(device).detach()
    mask, _ = gradcampp(_input)
    explanation = mask.to("cpu").detach().squeeze().numpy()
    return explanation

def RDE_explain(RDE_exp, image_t):
    num_iter = 1000000
    step_size = 1e-3
    batch_size = 64
    l1_lambda = 1000
    s, _ = generate_explainability_map(
        image_t, model, num_iter, step_size, batch_size, l1_lambda, device
    )
    s = s[0]
    explanation = s / np.max(s)
    return explanation

from skimage.segmentation import mark_boundaries

def init_lime():
    limexp = lime_image.LimeImageExplainer()
    return limexp

def lime_classifier(images):
    batch = torch.stack(tuple(transform(i) for i in images), dim=0)
    batch = batch.to(device)
    outputs = model(batch)
    logits = torch.nn.Softmax(dim=1)(outputs)
    return logits.detach().cpu().numpy()

def lime_explain(limexp, image_RGB):
    lime_input = np.array(image_RGB)
    lime_explanation = explainer.explain_instance(lime_input, lime_classifier, top_labels=2, hide_color=1, num_samples=200) 
    _, mask = lime_explanation.get_image_and_mask(lime_explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
    explanation = mask
    return explanation

exp_mapper = [
    {
        "name": "hexp/absolute_0",
        "init": init_hexp,
        "explain": hexp_explain
    },
    {
        "name": "hexp/relative_70",
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
        "init": lambda: GradCAM(model, model.conv2),
        "explain": gradcam_explain
    },
#     {
#         "name": "gradcampp",
#         "init": lambda: GradCAMpp(model, model.conv2),
#         "explain": gradcampp_explain
#     },
#     {
#         "name": "RDE",
#         "init": lambda: None,
#         "explain": RDE_explain
#     },
    {
        "name": "lime",
        "init": init_lime,
        "explain": lime_explain
    }
]

c = [6]
true_positives = np.load("true_positives.npy", allow_pickle=True).item()

comp_times = {}

for n in c:
    for exp in exp_mapper:
        exp_name = exp["name"]
        explainer = exp["init"]()
        explain = exp["explain"]
        print("Initialized %s" % exp_name)
        
        comp_times = []
        for i, image_path in enumerate(true_positives[str(n)]):
            image_name = os.path.basename(image_path)
            image = Image.open(image_path)
            image_RGB = image.convert("RGB")
            image_t = transform(image)
            if "hexp" in exp_name:
                _threshold = exp_name.split("/")[1].split("_")
                threshold_mode = _threshold[0]
                threshold_value = int(_threshold[1])
                threshold = threshold_value
                percentile = threshold_value
            t0 = time.time()
            if exp_name == "lime":
                explanation = explain(explainer, image_RGB)
            else:
                explanation = explain(explainer, image_t)
            torch.cuda.empty_cache()
            tf = time.time()
            runtime = round(tf - t0, 6)
            comp_times.append(runtime)
            print('%s: %d/%d(%d) runtime=%.4fs' % (exp_name, i+1, len(true_positives[str(n)]), n, runtime))
            np.save("true_positive_explanations/%s/%s" % (exp_name, image_name), explanation)

        np.save(os.path.join("true_positive_explanations", exp_name, f"comp_times_{n}.npy"), comp_times)
            