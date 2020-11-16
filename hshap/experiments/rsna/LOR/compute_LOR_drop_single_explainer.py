# NUMPY IMPORTS
import numpy as np
# PYTORCH IMPORTS
import torch 
import torch.nn as nn
from torchvision import models, datasets, transforms
# SYS IMPORTS
import GPUtil
import sys
import re
import os
# HSHAP IMPORTS
import hshap
# SHAP IMPORTS
import shap
# GRADCAM, GRADCAM++ IMPORTS
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

# READ ARGVS:
argvs = sys.argv
HOME = str(argvs[1])
EXP_SIZE = int(argvs[2])
REF_SIZE = int(argvs[3])
MIN_SIZE = int(argvs[4])
print(HOME, EXP_SIZE, REF_SIZE, MIN_SIZE)

# DEFINE GLOBAL CONSTANTS
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.299, 0.224, 0.225])
DATA_DIR = os.path.join(HOME, "repo/hshap/data/rsna/datasets")

# DEFINE DEVICE
_device = "cuda:0"
device = torch.device(_device)
torch.cuda.empty_cache()
print("Current device is {}".format(device))

# LOAD PRE-TRAINED INCEPTION-V3 MODEL
model = models.inception_v3(pretrained=True)
model.aux_logits = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
weight_path = os.path.join(HOME, "repo/hshap/data/rsna/pretrained-models/RSNA_InceptionV3.pth")
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
model.eval()
print("Loaded pretrained model")

RAIN_DATA_DIR = os.path.join(DATA_DIR, "train")

preprocess = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
)

# LOAD A RANDOM BATCH FROM THE TRAINING DATASET,
# THIS WILL BE FED TO SHAP METHODS TO EVALUATE THE REFERENCE
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")

preprocess = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
)

batch_size = REF_SIZE
train_data = datasets.ImageFolder(TRAIN_DATA_DIR, transform=preprocess)
dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers = 0)
train_loader = iter(dataloader)
X, _ = next(train_loader)
# X = X.detach().to(device)
ref = torch.mean(X, axis=0)
print("Loaded reference batch for shap methods to cpu")

deepexp = shap.DeepExplainer(model, X.to(device))
print("Initiliazed DeepExplainer on GPU")

def deepexp_explain(deepexp, input):
    deepexp_shapley_values, deepexp_indexes = deepexp.shap_values(input, ranked_outputs=2)
    deepexp_saliency = deepexp_shapley_values[0][0].sum(0)
    return deepexp_saliency

# LOAD SICK IMAGES FOR EXPERIMENT
EXP_DIR = os.path.join(HOME, "repo/hshap/data/rsna/LOR/datasets/{}".format(EXP_SIZE))
exp_img_dataset = hshap.utils.RSNASickDataset(EXP_DIR, preprocess)
exp_size = EXP_SIZE
exp_img_loader = torch.utils.data.DataLoader(exp_img_dataset, batch_size=exp_size, num_workers=0)
exp_iter = iter(exp_img_loader)
exp_imgs = next(exp_iter)
exp_imgs = exp_imgs
exp_imgs_L = len(exp_imgs)

print("----------")
print("Loaded {} images to CPU".format(exp_imgs_L))
print("----------")
GPUtil.showUtilization()

# DEFINE PERTURBATION SIZES
last_added = 0
exp_x = np.linspace(-1, 0, 20)
perturbation_sizes = np.sort(1.1 - 10**(exp_x))
perturbations_L = len(perturbation_sizes)
LOR = np.zeros((exp_imgs_L, perturbations_L))

# FOR EACH IMAGE
for eximg_id, image in enumerate(exp_imgs):
    
    print('Analyzing image #%d' % (eximg_id + 1))
    input = image.view(-1, 3, 299, 299).detach().to(device)
    # id_match = re.search("ex\d*_(\d*).png", image_name)
    # id = int(id_match.group(1))
    
    torch.cuda.empty_cache()
    saliency_map = deepexp_explain(deepexp, input)
    del input
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    
    activation_threshold = 0
    salient_points = np.where(saliency_map > activation_threshold)
    salient_rows = salient_points[0]
    salient_columns = salient_points[1]
    L = len(salient_rows)
    ids = np.arange(L)
        
    # PERTURBATE IMAGES AND EVALUATE LOR
    for k, perturbation_size in enumerate(perturbation_sizes):
        print("Perturbation={}".format(perturbation_size))
        perturbed_img = image.clone().to(device)
        perturbation_L = int(perturbation_size * L)
        perturbed_ids = np.random.choice(ids, replace = False, size = perturbation_L)
        perturbed_rows = salient_rows[perturbed_ids]
        perturbed_columns = salient_columns[perturbed_ids]
        
        for j in np.arange(perturbation_L):
            row = perturbed_rows[j]
            column = perturbed_columns[j]
            perturbed_img[:, row, column] = ref[:, row, column]
    
        perturbed_input = perturbed_img.view(-1, 3, 299, 299)
        prediction = model(perturbed_input).cpu().detach().numpy()[0]
        del perturbed_img
        torch.cuda.empty_cache()
        
        logits = np.exp(prediction)/np.sum(np.exp(prediction))

        if np.isnan(logits[1]):
            logits[1] = 1
        LOR[last_added, k] = np.log10(logits[1])