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
EXPL_ID = int(argvs[5])
GPU_ID = int(argvs[6])


# DEFINE GLOBAL CONSTANTS
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.299, 0.224, 0.225])
if HOME == "gaon":
    DATA_DIR = "/export/gaon1/data/jteneggi/data/rsna/datasets"
else:
    DATA_DIR = os.path.join(HOME, "repo/hshap/data/rsna/datasets")

# DEFINE DEVICE
_device = "cuda:{}".format(GPU_ID)
device = torch.device(_device)
torch.cuda.empty_cache()
print("Current device is {}".format(device))

# LOAD PRE-TRAINED INCEPTION-V3 MODEL
model = models.inception_v3(pretrained=True)
model.aux_logits = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
if HOME  == "gaon":
    weight_path = "/export/gaon1/data/jteneggi/data/rsna/pretrained-models/RSNA_InceptionV3.pth"
else:
    weight_path = os.path.join(HOME, "repo/hshap/data/rsna/pretrained-models/RSNA_InceptionV3.pth")
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
model.eval()
print("Loaded pretrained model")

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
X = X.detach().to(device)
ref = torch.mean(X, axis=0)
print("Loaded reference batch for shap methods")

# INITIALIZE EXPLAINERS
if EXPL_ID == 0:
    explainer = shap.GradientExplainer(model, X)
if EXPL_ID == 1:
    explainer = shap.DeepExplainer(model, X)
if EXPL_ID == 2:
    explainer = hshap.src.Explainer(model, ref)
if EXPL_ID == 3:
    explainer = GradCAM(model, model.Mixed_7c)
if EXPL_ID == 4:
    explainer = GradCAMpp(model, model.Mixed_7c)
# LOAD HSHAP REFERENCE
# REFERENCE_PATH = os.path.join(HOME, "repo/hshap/data/rsna/references/avg_all_train.npy")
# print("Loaded reference for hshap")
# np_ref = np.load(REFERENCE_PATH)
# ref = torch.from_numpy(np_ref).to(device)
# DEFINE CAMS (take the last but conv layer)

# DEFINE EXPLAINER FUNCTIONS FOR EACH EXPLAINER
def gradexp_explain(gradexp, input, ranked_outputs = 2, nsamples = 200):
    gradexp_shapley_values, gradexp_indexes = gradexp.shap_values(input, ranked_outputs=2, nsamples=nsamples)
    gradexp_saliency = gradexp_shapley_values[0][0].sum(0)
    return gradexp_saliency

def deepexp_explain(deepexp, input):
    deepexp_shapley_values, deepexp_indexes = deepexp.shap_values(input, ranked_outputs=2)
    deepexp_saliency = deepexp_shapley_values[0][0].sum(0)
    return deepexp_saliency

def hexp_explain(hexp, image, threshold, minW, minH, label=1):
    hexp_saliency, flatnodes = hexp.explain(
        image, label=label, threshold=threshold, minW=minW, minH=minH
    )
    return hexp_saliency

def gradcam_explain(gradcam, input):
    mask, _ = gradcam(input)
    gradcam_saliency = mask.cpu().detach().squeeze().numpy()
    return gradcam_saliency

def gradcampp_explain(gradcampp, input):
    maskpp, _ = gradcampp(input)
    gradcampp_saliency = maskpp.cpu().detach().squeeze().numpy()
    return gradcampp_saliency

# INITIALIZE EXPLAINERS DICTIONARY
explainer_dictionary = {
  0: {'name': 'Grad-Explainer', 'explain': gradexp_explain },
  1: {'name': 'Deep-Explainer', 'explain': deepexp_explain},
  2: {'name': 'H-Explainer', 'explain': hexp_explain},
  3: {'name': 'GradCAM', 'explain': gradcam_explain},
  4: {'name': 'GradCAM++', 'explain': gradcampp_explain},
}
# explainer = explainer_dictionary[EXPL_ID]
# explainers_L = len(explainer_dictionary) # number of explainers
# last_added = np.zeros((explainers_L), dtype=np.uint16)

# LOAD SICK IMAGES FOR EXPERIMENT
if HOME == "gaon":
    EXP_DIR = "/export/gaon1/data/jteneggi/data/rsna/LOR/datasets/{}".format(EXP_SIZE)
else:
    EXP_DIR = os.path.join(HOME, "repo/hshap/data/rsna/LOR/datasets/{}".format(EXP_SIZE))
exp_img_dataset = hshap.utils.RSNASickDataset(EXP_DIR, preprocess)
exp_size = EXP_SIZE
exp_img_loader = torch.utils.data.DataLoader(exp_img_dataset, batch_size=exp_size, num_workers=0)
exp_iter = iter(exp_img_loader)
exp_imgs = next(exp_iter)
exp_imgs = exp_imgs.to(device)
exp_imgs_L = len(exp_imgs)

# LOAD EXAMPLE IMAGES FOR EXPERIMENT
# ImF = ImageFolder(os.path.join(HOME, 'data/Jacopo/HShap/LOR'), transform = transf)
# ImF_names = ImF.imgs
# example_batch_size = 300
# exloader = DataLoader(ImF, batch_size = example_batch_size, shuffle = False, num_workers = 0)
# exIter = iter(exloader)
# images, _ = next(exIter)
# EXIMG_L = len(images) 

print("----------")
print("Loaded {} images".format(exp_imgs_L))
print("----------")


# DEFINE EXPLAINER METHOD
explainer_name = explainer_dictionary[EXPL_ID]['name']
explain = explainer_dictionary[EXPL_ID]['explain']
print("Initialized {} explainer".format(explainer_name))


# DEFINE PERTURBATION SIZES
exp_x = np.linspace(-1, 0, 20)
perturbation_sizes = np.sort(1.1 - 10**(exp_x))
perturbations_L = len(perturbation_sizes)
LOR = torch.zeros((exp_imgs_L, perturbations_L)).to(device)

# FOR EACH IMAGE
for eximg_id, image in enumerate(exp_imgs):
    
    print('Analyzing image #%d' % (eximg_id + 1))
    img_t0 = time.time()
    input = image.view(-1, 3, 299, 299).detach()
    # id_match = re.search("ex\d*_(\d*).png", image_name)
    # id = int(id_match.group(1))
        
    # COMPUTE SALIENCY MAPS
    exp_t0 = time.time()
    if explainer_name == 'Grad-Explainer':
        torch.cuda.empty_cache()
        saliency_map = explain(explainer, input)
        
    elif explainer_name == 'Deep-Explainer':
        torch.cuda.empty_cache()
        saliency_map = explain(explainer, input)
        
    elif explainer_name == 'H-Explainer':
        torch.cuda.empty_cache()
        threshold = 0
        minSize = MIN_SIZE
        label = 1
        saliency_map = explain(explainer, image, threshold=threshold, minW=minSize, minH=minSize, label=label)

    elif explainer_name == 'GradCAM' or explainer_name == 'GradCAM++':
        torch.cuda.empty_cache()
        saliency_map = explain(explainer, input)
    
    print("Explanation time for %s %.3f" % (explainer_name, time.time() - exp_t0))

    # EXTRACT SALIENT POINTS FROM SALIENCY MAP
    activation_threshold = 0
    salient_points = np.where(saliency_map > activation_threshold)
    salient_rows = salient_points[0]
    salient_columns = salient_points[1]
    L = len(salient_rows)
    ids = np.arange(L)
        
    # PERTURBATE IMAGES AND EVALUATE LOR
    # DEFINE PERTURBED INPUT BATCH
    perturbed_batch = image.view(-1, 3, 299, 299).repeat(perturbations_L, 1, 1, 1)
    for k, perturbation_size in enumerate(perturbation_sizes):
        print("Perturbation={}".format(perturbation_size))
        
        perturbation_L = int(perturbation_size * L)
        perturbed_ids = np.random.choice(ids, replace = False, size = perturbation_L)
        perturbed_rows = salient_rows[perturbed_ids]
        perturbed_columns = salient_columns[perturbed_ids]
        
        for j in np.arange(perturbation_L):
            row = perturbed_rows[j]
            column = perturbed_columns[j]
            perturbed_batch[k, :, row, column] = ref[:, row, column]
    
    prediction = model(perturbed_batch).detach()
    del perturbed_batch
    torch.cuda.empty_cache()
    with torch.no_grad():
        aux = torch.exp(prediction)
        logits = torch.div(aux[:, 1], torch.sum(aux, 1))
        logits = torch.log10(logits)
                                                         
    LOR[eximg_id, :] = logits
    print("Analyzed image {} in {}s".format(eximg_id + 1, time.time() - img_t0))

if HOME == "gaon":
    np.save(os.path.join("/cis/home/jteneggi", 'repo/hshap/hshap/experiments/rsna/LOR/results/_{}_{}_{}_{}.npy'.format(explainer_name, EXP_SIZE, REF_SIZE, MIN_SIZE)), LOR.cpu().numpy())
else:
    np.save(os.path.join(HOME, 'repo/hshap/hshap/experiments/rsna/LOR/results/_{}_{}_{}_{}.npy'.format(explainer_name, EXP_SIZE, REF_SIZE, MIN_SIZE)), LOR.cpu().numpy())
