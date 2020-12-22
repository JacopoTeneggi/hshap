import numpy as np
from numpy.random import choice
import os
import torch
import pandas as pd
from shutil import copyfile

DATA_DIR = "/export/gaon1/data/jteneggi/data/malaria/"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")
TROPHOZOITE_DIR = os.path.join(DATA_DIR, "trophozoite")
TROP_TRAIN_DIR = os.path.join(TROPHOZOITE_DIR, "train")
TROP_TEST_DIR = os.path.join(TROPHOZOITE_DIR, "test")
TROP_VAL_DIR = os.path.join(TROPHOZOITE_DIR, "val")

print(os.path.join(DATA_DIR, "training.json"))
df_training = pd.read_json(os.path.join(DATA_DIR, "training.json"))
df_test = pd.read_json(os.path.join(DATA_DIR, "test_cropped.json"))

class_dict = {
    "red blood cell": 0,
    "leukocyte": 1,
    "gametocyte": 2,
    "ring": 3,
    "trophozoite": 4,
    "schizont": 5,
    "difficult": 6
}

L = len(df_training)
q = len(class_dict)
cells_count = torch.zeros((L, q))

# DEFINE DICTIONARY CONTAINING IMAGES BOTH IN THE TRAINING AND TEST SETS THAT
# HAVE OR DO NOT HAVE AT LEAST ONE TROPHOZOITE
trophozoites = {0:[], 1:[]}
for df in [df_training, df_test]:
    for i, image in df_training.iterrows():
        image_name = os.path.basename(image["image"]["pathname"])
        cells = image["objects"]
        for cell in cells:
            cell_class = cell["category"]
            cells_count[i, class_dict[cell_class]] += 1

        if cells_count[i, class_dict["trophozoite"]] >= 1:
            trophozoites[1].append(image_name)
        else:
            trophozoites[0].append(image_name)

# RANDOMLY CHOOSE 120 IMAGES PER CLASS FROM THE TRAINING DATASET TO BE IN THE TEST SET
train_trophozoite = {0:[], 1:[]}
test_trophozoite = {0:[], 1:[]}
test_trophozoite_0_ids = choice(len(trophozoites[0]), size=120, replace=False)
test_trophozoite_1_ids = choice(len(trophozoites[1]), size=120, replace=False)

# SPLIT TRAIN AND TEST SETS
for i in test_trophozoite_0_ids:
    test_trophozoite[0].append(trophozoites[0][i])
trophozoites[0] = np.delete(trophozoites[0], test_trophozoite_0_ids)
for i in test_trophozoite_1_ids:
    test_trophozoite[1].append(trophozoites[1][i])
trophozoites[1] = np.delete(trophozoites[1], test_trophozoite_1_ids)

# COPY TRAIN AND TEST SPLITS
for _class in [0, 1]:
    for i, image in enumerate(trophozoites[_class]):
        print("copied {} into train/{}".format(i, _class))
        copyfile(os.path.join(IMG_DIR, image), os.path.join(TROP_TRAIN_DIR, "{}/{}".format(str(_class), image)))
    for i, image in enumerate(test_trophozoite[_class]):
        print("copied {} into test/{}".format(i, _class))
        copyfile(os.path.join(IMG_DIR, image), os.path.join(TROP_TEST_DIR, "{}/{}".format(str(_class), image)))

# i = len(df_test)
# q = len(class_dict)
# test_cells_count = torch.zeros((L, q))
# test_ring = {0:[], 1:[]}
# test_trophozoite = {0:[], 1:[]}
# for i, image in df_test.iterrows():
#     image_name = os.path.basename(image["image"]["pathname"])
#     cells = image["objects"]
#     for cell in cells:
#         cell_class = cell["category"]
#         test_cells_count[i, class_dict[cell_class]] += 1
# 
#     if test_cells_count[i, class_dict["trophozoite"]] >= 1:
#         test_trophozoite[1].append(image_name)
#         copyfile(os.path.join(IMG_DIR, image_name), os.path.join(TROP_TEST_DIR, "{}/{}".format(1, image_name))) 
#     else:
#         test_trophozoite[0].append(image_name)
#         copyfile(os.path.join(IMG_DIR, image_name), os.path.join(TROP_TEST_DIR, "{}/{}".format(0, image_name)))
#
#     if test_cells_count[i, class_dict["ring"]] >= 1:
#         test_ring[1].append(image_name)
#     else:
#         test_ring[0].append(image_name)

print("Count done")
