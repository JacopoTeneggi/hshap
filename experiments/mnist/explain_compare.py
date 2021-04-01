from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
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
# from RDE.ComputeExplainability import generate_explainability_map 
from lime import lime_image

os.environ["CUDA_VISIBLE_DEVICES"]="7"

device = torch.device("cuda:0")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
torch.manual_seed(0)
model = Net()
weight_path = "mnist_cnn.pt"
model.load_state_dict(torch.load(weight_path, map_location=device)) 
model.to(device)
model.eval()
    
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST("/export/gaon1/data/jteneggi/data/mnist/data", train=True, transform=transform)
test_dataset = datasets.MNIST("/export/gaon1/data/jteneggi/data/mnist/data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

def init_hexp():
    _iter = iter(train_loader)
    X, _ = next(_iter)
    ref = X.detach().mean(0)
    ref = ref.to(device)
    hexp = hshap.src.Explainer(model, ref)
    return hexp

def hexp_explain(hexp, image_t):
    (explanation, _) = hexp.explain(image_t, label=digit, minW=5, minH=5, threshold_mode=threshold_mode, percentile=percentile, threshold=threshold)
    return explanation

def init_gradexp():
    _iter = iter(train_loader)
    X, _ = next(_iter)
    X = X.to(device)
    gradexp = shap.GradientExplainer(model, X)
    return gradexp

def gradexp_explain(gradexp, image_t):
    _input = image_t.unsqueeze(0).to(device).detach()
    gradexp_shapley_values, gradexp_indexes = gradexp.shap_values(_input, ranked_outputs=2, nsamples=1000)
    explanation = gradexp_shapley_values[0][0].sum(0)
    return explanation

def init_deepexp():
    _iter = iter(train_loader)
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
    masker = TensorImageMasker("inpaint_telea", (28, 28))   
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
#     {
#         "name": "partexp",
#         "init": init_partexp,
#         "explain": partexp_explain
#     },
    {
        "name": "gradcam",
        "init": lambda: GradCAM(model, model.conv2),
        "explain": gradcam_explain
    }
#     ,
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
#     {
#         "name": "lime",
#         "init": init_lime,
#         "explain": lime_explain
#     }
]

for exp in exp_mapper[:2]:
    true_positive_id = 0
    exp_name = exp["name"]
    explainer = exp["init"]()
    explain = exp["explain"]
    print(f"Initialized %s" % exp_name)
    
    comp_times = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        digit = output.argmax(dim=1, keepdim=True).item()
        if digit == target:
            if true_positive_id == 10:
                break
            if "hexp" in exp_name:
                _threshold = exp_name.split("/")[1].split("_")
                threshold_mode = _threshold[0]
                threshold_value = int(_threshold[1])
                threshold = 0.1
                percentile = threshold_value
            data = data.squeeze(0)
            t0 = time.time()
            explanation = explain(explainer, data)
            tf = time.time()
            torch.cuda.empty_cache()
            runtime = round(tf - t0, 6)
            comp_times.append(runtime)
            print('%s: %d runtime=%.4fs' % (exp_name, true_positive_id, runtime))
            np.save("true_positive_explanations/%s/%s" % (exp_name, true_positive_id), explanation)
        true_positive_id += 1
    np.save(os.path.join("true_positive_explanations", exp_name, "comp_times.npy"), comp_times) 
                