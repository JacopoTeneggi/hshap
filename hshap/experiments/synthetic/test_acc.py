import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import scipy
import os
import pandas as pd
from hshap.utils import Net

os.environ["CUDA_VISIBLE_DEVICES"]="9"

device = torch.device("cuda:0")

torch.manual_seed(0)
model = Net()
weight_path = "model2.pth"
model.load_state_dict(torch.load(weight_path, map_location=device)) 
model.to(device)
model.eval()

# df_training = pd.read_json("/export/gaon1/data/jteneggi/data/malaria/training.json")
# df_test = pd.read_json("/export/gaon1/data/jteneggi/data/malaria/test_cropped.json")
# frames = [df_training, df_test]
# df_merged = pd.concat(frames, ignore_index=True)
# ADD IMAGE_NAME COLUMN TO DATAFRAME
# image_names = []
# for i, row in df_merged.iterrows():
#     image_names.append(os.path.basename(row["image"]["pathname"]))
# df_merged["image_name"] = image_names

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_dir = "/export/gaon1/data/jteneggi/data/synthetic/LOR"
dataset = datasets.ImageFolder(os.path.join(data_dir, "images"), transform)
image_names = [os.path.basename(sample[0]) for sample in dataset.samples]
L = len(image_names)
print("Found %d test images" % L)
true_positives = {}
n = 3
classes = np.arange(1, 10)
for c in classes:
    true_positives[str(c)] = []
print(true_positives)
false_negatives = []
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)
for batch_id, (images, labels) in enumerate(dataloader):
    images = images.to(device)
    labels = labels.to(device)
    # print(images.size())
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    np_labels = labels.cpu().numpy()
    np_preds = preds.detach().cpu().numpy()
    print(np_labels)
    
    for i, label in enumerate(np_labels):
        image_id = batch_id * 4 + i
        image_name = image_names[image_id]
        crosses_count = classes[label]
        image = os.path.join(data_dir, "images/{}/{}".format(crosses_count, image_name))
        if np_preds[i] == 1:
            true_positives[str(crosses_count)].append(image)
        if np_preds[i] == 0:
            false_negatives.append(image)

correct = 0
wrong = 0
for c in true_positives:
    print(c, len(true_positives[c]))
    correct += len(true_positives[c])
wrong += len(false_negatives)
accuracy = correct/(correct+wrong)
print("Test accuracy: %.2f" % accuracy)
np.save("true_positives", true_positives, allow_pickle=True)
print("Saved true positives")