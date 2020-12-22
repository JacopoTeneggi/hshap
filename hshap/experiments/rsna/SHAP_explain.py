import numpy as np
import torch
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import cv2
import HShap as HShap
from PIL import Image
import re

# DEFINE GLOBAL CONSTANTS
HOME = "."

# IMPORT HSHAP
import HShap as HShap

# IMPORT SHAP
import shap

# INSTALL GRADCAM, GRADCAM++
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

# INSTALL KERAS DEPS FOR KERNEL-SHAP
from skimage.segmentation import slic
from keras.preprocessing import image as keras_img_tools

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

avg_all_train = np.load("avg_all_train.npy")
background = torch.from_numpy(avg_all_train).detach()

# # LOAD TRAIN DATA
# MEAN, STD = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
# SYNTH_DATA_DIR = os.path.join(HOME, "data/Jacopo/HShap/Synthetic_data/")
# transf = transforms.Compose([ToTensor(), Normalize(mean=MEAN, std=STD)])
# train_batch_size = 64
# train_data = ImageFolder(root=os.path.join(SYNTH_DATA_DIR, "train"), transform=transf)
# dataloader = DataLoader(
#     train_data, batch_size=train_batch_size, shuffle=True, num_workers=0
# )
# train_loader = iter(dataloader)
# X, Y = next(train_loader)
# # DEFINE MASK BACKGROUND WITH AVERAGE OF TRAINING SET
# background = torch.mean(X, dim=0)
# background.detach()

# INITIALIZE EXPLAINERS
gradexp = shap.GradientExplainer(model, X)
deepexp = shap.DeepExplainer(model, X)
hexp = HShap.Explainer(model, background)
# define cams
gradcam = GradCAM(model, model.conv2)
gradcampp = GradCAMpp(model, model.conv2)
# define KernelSHAP parameters
width = 120
height = 100
feature_width = 10
feature_height = 10
ROW_L = int(height / feature_height)
COLUMN_L = int(width / feature_width)

# DEFINE EXPLAINER FUNCTIONS FOR EACH EXPLAINER
def gradexp_explain(gradexp, input, ranked_outputs=2, nsamples=200):
    gradexp_shapley_values, gradexp_indexes = gradexp.shap_values(
        input, ranked_outputs=2, nsamples=nsamples
    )
    gradexp_saliency = gradexp_shapley_values[0][0].sum(0)
    return gradexp_saliency


def deepexp_explain(deepexp, input):
    deepexp_shapley_values, deepexp_indexes = deepexp.shap_values(
        input, ranked_outputs=2
    )
    deepexp_saliency = deepexp_shapley_values[0][0].sum(0)
    return deepexp_saliency


def hexp_explain(hexp, image, threshold, minW, minH, label=1):
    hexp_saliency, flatnodes = hexp.explain(
        image, label=label, threshold=threshold, minW=minW, minH=minH
    )
    return hexp_saliency


def gradcam_explain(gradcam, input):
    mask, _ = gradcam(input)
    gradcam_saliency = mask.detach().squeeze().numpy()
    return gradcam_saliency


def gradcampp_explain(gradcampp, input):
    maskpp, _ = gradcampp(input)
    gradcampp_saliency = maskpp.detach().squeeze().numpy()
    return gradcampp_saliency


# DEFINE HELPER FUNCTIONS FOR KERNEL-SHAP
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background
    # SHIFT CHANNELS: form (1, 100, 120, 3) to (1, 3, 100, 120)
    out2 = np.zeros(
        (zs.shape[0], image.shape[2], image.shape[0], image.shape[1]), dtype="d"
    )
    for i in range(zs.shape[0]):
        for j in range(image.shape[2]):
            out2[i, j, :, :] = out[i, :, :, j]
    out2 = torch.from_numpy(out2)
    out2 = out2.type(torch.FloatTensor)
    return out2.detach()


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


def f(z):
    return model(mask_image(z, segments, img_orig, background=255)).detach().numpy()


def kernelexp_explain(kernelxp, segments):
    kernelexp_shapley_values = kernelexp.shap_values(
        np.ones((1, ROW_L * COLUMN_L)), nsamples=1000
    )
    kernelexp_saliency = fill_segmentation(kernelexp_shapley_values[1][0], segments)
    return kernelexp_saliency


# INITIALIZE EXPLAINERS DICTIONARY
expmapper = {
    0: {"name": "Grad-Explainer", "exp": gradexp, "explain": gradexp_explain},
    1: {"name": "Deep-Explainer", "exp": deepexp, "explain": deepexp_explain},
    2: {"name": "H-Explainer", "exp": hexp, "explain": hexp_explain},
    3: {"name": "GradCAM", "exp": gradcam, "explain": gradcam_explain},
    4: {"name": "GradCAM++", "exp": gradcampp, "explain": gradcampp_explain},
    5: {"name": "Kernel-SHAP", "exp": None, "explain": kernelexp_explain},
}
EXP_L = len(expmapper)  # number of explainers
LAST_ADDED = np.zeros((3, EXP_L), dtype=np.uint16)

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

HOME = "."
DATA_DIR = os.path.join(HOME, "RSNA_example_images")
example_set = datasets.ImageFolder(DATA_DIR)
example_names = example_set.imgs
batch_size = 4
example_loader = torch.utils.data.DataLoader(
    example_set, batch_size=batch_size, shuffle=False, num_workers=0
)
example_iter = iter(example_loader)

avg_all_train = np.load("avg_all_train.npy")
background = torch.from_numpy(avg_all_train)

hexp = HShap.Explainer(model, background, M=4)
data = next(example_iter)
images, _ = data
image = image[0]

hexp_saliency = hexp.explain(image, label=1, threshold=0, minW=10, minH=10)
