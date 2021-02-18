import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Normalize
from torchvision import transforms
import re
import os

# DEFINE GLOBAL CONSTANTS
HOME = "/home-net/home-2/jtenegg1@jhu.edu"

# IMPORT UTILS
import local_modules.utils as Utils

# IMPORT HSHAP
from local_modules.HShap import Explainer as HShapExplainer

# INSTALL SHAP
import shap

# INSTALL GRADCAM, GRADCAM++
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

# INSTALL KERAS DEPS FOR KERNEL-SHAP
from skimage.segmentation import slic
from keras.preprocessing import image as keras_img_tools

# LOAD PRE-TRAINED-NETWORK
model = Utils.Net()
model.load_state_dict(
    torch.load(os.path.join(HOME, "data/Jacopo/HShap/Pretrained_models/model2.pth"))
)
# model.eval() deactivates the dropout layer in the network
model.eval()

# LOAD TRAIN DATA
MEAN, STD = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
SYNTH_DATA_DIR = os.path.join(HOME, "data/Jacopo/HShap/Synthetic_data/")
transf = transforms.Compose([ToTensor(), Normalize(mean=MEAN, std=STD)])
train_batch_size = 64
train_data = ImageFolder(root=os.path.join(SYNTH_DATA_DIR, "train"), transform=transf)
dataloader = DataLoader(
    train_data, batch_size=train_batch_size, shuffle=True, num_workers=0
)
train_loader = iter(dataloader)
X, Y = next(train_loader)
# DEFINE MASK BACKGROUND WITH AVERAGE OF TRAINING SET
background = torch.mean(X, dim=0)
background.detach()

# INITIALIZE EXPLAINERS
gradexp = shap.GradientExplainer(model, X)
deepexp = shap.DeepExplainer(model, X)
hexp = HShapExplainer(model, background)
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

# LOAD EXAMPLE IMAGES FOR EXPERIMENT
ImF = ImageFolder(os.path.join(HOME, "data/Jacopo/HShap/LOR"), transform=transf)
ImF_names = ImF.imgs
example_batch_size = 300
exloader = DataLoader(ImF, batch_size=example_batch_size, shuffle=False, num_workers=0)
exIter = iter(exloader)
images, _ = next(exIter)
EXIMG_L = len(images)

print("----------")
print("Loaded %d images" % EXIMG_L)
print("----------")

# DEFINE PERTURBATION SIZES
perturbation_sizes = np.linspace(0, 1, 100)
LOR_L = len(perturbation_sizes)
classes_n = 3  # 3 is for the known number of crosses in an image: 1, 2, or 3
LOR = np.zeros((3, EXP_L, int(EXIMG_L / 3), LOR_L))

# FOR EACH IMAGE
for eximg_id in np.arange(EXIMG_L):

    print("Analyzing image #%d" % (eximg_id + 1))
    image = images[eximg_id]

    # Extract number of crosses in image
    image_name = ImF_names[eximg_id][0]
    crosses_match = re.search("ex(\d*)_\d*.png", image_name)
    id_match = re.search("ex\d*_(\d*).png", image_name)

    if crosses_match is None or id_match is None:
        print("ERROR %s" % image_name)
        continue

    crosses = int(crosses_match.group(1))
    id = int(id_match.group(1))

    # FOR EACH EXPLAINER
    for exp_id in np.arange(EXP_L):

        # READ EXPLAINER
        exp_name = expmapper[exp_id]["name"]
        exp = expmapper[exp_id]["exp"]
        explain = expmapper[exp_id]["explain"]

        # COMPUTE SALIENCY MAPS
        if exp_name == "Grad-Explainer":

            input = image.view(-1, 3, 100, 120).detach()
            saliency_map = explain(exp, input)

        elif exp_name == "Deep-Explainer":

            input = image.view(-1, 3, 100, 120).detach()
            saliency_map = explain(exp, input)

        elif exp_name == "H-Explainer":

            threshold = 0
            minSize = 2
            label = 1
            saliency_map = explain(
                exp, image, threshold=threshold, minW=minSize, minH=minSize, label=label
            )
        elif exp_name == "GradCAM" or exp_name == "GradCAM++":
            input = image.view(-1, 3, 100, 120).detach()
            saliency_map = explain(exp, input)

        elif exp_name == "Kernel-SHAP":

            # LOAD IMAGE IN KERAS
            img = keras_img_tools.load_img(image_name)
            img_orig = keras_img_tools.img_to_array(img)

            # DEFINE SEGMENTS
            segments = np.zeros((height, width))
            segment_id = 0
            for segment_row_id in np.arange(ROW_L + 1):
                startRow = int((segment_row_id - 1) * height)
                endRow = int(segment_row_id * height)
                for segment_column_id in np.arange(COLUMN_L + 1):
                    startColumn = int((segment_column_id - 1) * width)
                    endColumn = int(segment_column_id * width)
                    segments[startRow:endRow, startColumn:endColumn] = segment_id
                    segment_id += 1

            # DEFINE KERNEL-SHAP
            kernelexp = shap.KernelExplainer(f, np.zeros((1, ROW_L * COLUMN_L)))

            saliency_map = explain(kernelexp, segments)

        activation_threshold = 0
        salient_points = np.where(saliency_map > activation_threshold)
        salient_rows = salient_points[0]
        salient_columns = salient_points[1]
        L = len(salient_rows)
        ids = np.arange(L)

        # PERTURBATE IMAGES AND EVALUATE LOR
        for k in np.arange(LOR_L):

            perturbation_size = perturbation_sizes[k]
            perturbation_L = int(perturbation_size * L)
            perturbed_ids = np.random.choice(ids, replace=False, size=perturbation_L)

            perturbed_img = image.clone()
            perturbed_rows = salient_rows[perturbed_ids]
            perturbed_columns = salient_columns[perturbed_ids]

            for j in np.arange(perturbation_L):
                row = perturbed_rows[j]
                column = perturbed_columns[j]
                perturbed_img[:, row, column] = background[:, row, column]

            perturbed_input = perturbed_img.view(-1, 3, 100, 120)
            prediction = model(perturbed_input).detach().numpy()[0]

            logits = np.exp(prediction) / np.sum(np.exp(prediction))

            if np.isnan(logits[1]):
                logits[1] = 1
            LOR[crosses - 1, exp_id, LAST_ADDED[crosses - 1, exp_id], k] = np.log10(
                logits[1]
            )

        print(
            "Saved at %d, %d, %d"
            % (exp_id, crosses - 1, LAST_ADDED[crosses - 1, exp_id])
        )
        LAST_ADDED[crosses - 1, exp_id] += 1

np.save(os.path.join(HOME, "data/Jacopo/HShap/LOR/perturbation"), LOR)
