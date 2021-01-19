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

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

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

# LOAD A RANDOM BATCH FROM THE TRAINING DATASET,
# THIS WILL BE FED TO SHAP METHODS TO EVALUATE THE REFERENCE
# TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize([0.7206, 0.7204, 0.7651], [0.2305, 0.2384, 0.1706])
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

# batch_size = REF_SIZE
# train_data = datasets.ImageFolder(TRAIN_DATA_DIR, transform=preprocess)
# dataloader = torch.utils.data.DataLoader(
#     train_data, batch_size=batch_size, shuffle=True, num_workers=0
# )
# train_loader = iter(dataloader)
# X, _ = next(train_loader)
# X = X.detach().to(device)
# ref = torch.mean(X, axis=0)
# print("Loaded reference batch for shap methods")

# INITIALIZE EXPLAINERS
# if EXPL_ID == 0:
#     explainer = shap.GradientExplainer(model, X)
# if EXPL_ID == 1:
#     explainer = shap.DeepExplainer(model, X)
# if EXPL_ID == 2:
#    explainer = hshap.src.Explainer(model, ref)
# if EXPL_ID == 3:
#     explainer = GradCAM(model, model.Mixed_7c)
# if EXPL_ID == 4:
#     explainer = GradCAMpp(model, model.Mixed_7c)
# LOAD HSHAP REFERENCE
# REFERENCE_PATH = os.path.join(HOME, "repo/hshap/data/rsna/references/avg_all_train.npy")
# print("Loaded reference for hshap")
# np_ref = np.load(REFERENCE_PATH)
# ref = torch.from_numpy(np_ref).to(device)
# DEFINE CAMS (take the last but conv layer)

# DEFINE EXPLAINER FUNCTIONS FOR EACH EXPLAINER
# def gradexp_explain(gradexp, input, ranked_outputs=2, nsamples=200):
#     gradexp_shapley_values, gradexp_indexes = gradexp.shap_values(
#         input, ranked_outputs=2, nsamples=nsamples
#     )
#     gradexp_saliency = gradexp_shapley_values[0][0].sum(0)
#     return gradexp_saliency


# def deepexp_explain(deepexp, input):
#     deepexp_shapley_values, deepexp_indexes = deepexp.shap_values(
#         input, ranked_outputs=2
#     )
#     deepexp_saliency = deepexp_shapley_values[0][0].sum(0)
#     return deepexp_saliency


# def hexp_explain(hexp, image, threshold, minW, minH, label=1):
#     hexp_saliency, flatnodes = hexp.explain(
#         image, label=label, threshold=threshold, minW=minW, minH=minH
#     )
#     return hexp_saliency


# def gradcam_explain(gradcam, input):
#     mask, _ = gradcam(input)
#    gradcam_saliency = mask.cpu().detach().squeeze().numpy()
#     return gradcam_saliency


# def gradcampp_explain(gradcampp, input):
#     maskpp, _ = gradcampp(input)
#     gradcampp_saliency = maskpp.cpu().detach().squeeze().numpy()
#     return gradcampp_saliency


# INITIALIZE EXPLAINERS DICTIONARY
exp_mapper = ["hexp/absolute_0", "gradexp", "deepexp", "partexp", "gradcam", "gradcampp", "naive", "RDE", "lime"]
# explainer = explainer_dictionary[EXPL_ID]
# explainers_L = len(explainer_dictionary) # number of explainers
# last_added = np.zeros((explainers_L), dtype=np.uint16)

# LOAD SICK IMAGES FOR EXPERIMENT
# if HOME == "gaon":
#     EXP_DIR = "/export/gaon1/data/jteneggi/data/rsna/LOR/datasets/{}".format(EXP_SIZE)
# else:
#     EXP_DIR = os.path.join(
#         HOME, "repo/hshap/data/rsna/LOR/datasets/{}".format(EXP_SIZE)
#     )
# exp_img_dataset = hshap.utils.RSNASickDataset(EXP_DIR, preprocess)
# exp_size = EXP_SIZE
# exp_img_loader = torch.utils.data.DataLoader(
#     exp_img_dataset, batch_size=exp_size, num_workers=0
# )
# exp_iter = iter(exp_img_loader)
# exp_imgs = next(exp_iter)
# exp_imgs = exp_imgs.to(device)
# exp_imgs_L = len(exp_imgs)

# LOAD EXAMPLE IMAGES FOR EXPERIMENT
# ImF = ImageFolder(os.path.join(HOME, 'data/Jacopo/HShap/LOR'), transform = transf)
# ImF_names = ImF.imgs
# example_batch_size = 300
# exloader = DataLoader(ImF, batch_size = example_batch_size, shuffle = False, num_workers = 0)
# exIter = iter(exloader)
# images, _ = next(exIter)
# EXIMG_L = len(images)

# print("----------")
# print("Loaded {} images".format(exp_imgs_L))
# print("----------")


# DEFINE EXPLAINER METHOD
# explainer_name = explainer_dictionary[EXPL_ID]["name"]
# explain = explainer_dictionary[EXPL_ID]["explain"]
# print("Initialized {} explainer".format(explainer_name))


# DEFINE PERTURBATION SIZES
# exp_x = np.linspace(-1, 0, 20)
# perturbation_sizes = np.sort(1.1 - 10 ** (exp_x))
# perturbations_L = len(perturbation_sizes)
A = 100*120
exp_x = np.linspace(np.log10(1/A), 0, 100)
relative_perturbation_sizes = np.concatenate(([0], np.sort(10 ** (exp_x))))
perturbation_sizes = np.round(A * relative_perturbation_sizes)
perturbation_sizes = np.array(perturbation_sizes, dtype="int")
print(perturbation_sizes)
perturbations_L = len(perturbation_sizes)

c = np.arange(1, 10)
true_positives = np.load("true_positives.npy", allow_pickle=True)
for n in c:
    images = true_positives.item()[str(n)]
    L = len(images)
    for exp_name in exp_mapper[-2:]:
        exp_logits = torch.zeros((L, perturbations_L)).to(device)
        explanation_dir = os.path.join("true_positive_explanations", exp_name)
        for i, image_path in enumerate(images):
            image_name = os.path.basename(image_path)
            image = transform(Image.open(image_path)).to(device).detach()
            if exp_name == "naive":
                explanation = torch.rand(image.size(1), image.size(2), device=torch.device("cpu")) + .5
            else:
                explanation = np.load(os.path.join(explanation_dir, "%s.npy" % image_name))
            exp_logits[i, :] = hshap.utils.compute_perturbed_logits(model, ref, image, explanation, perturbation_sizes, normalization="original")
            print("%s: %d/%d computed perturbed logits" % (exp_name, i+1, L))
        np.save(os.path.join("LOR", "%s/results_%d" % (exp_name, n)), exp_logits.cpu().numpy())
    
