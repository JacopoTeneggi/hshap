import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import scipy
import os
import pandas as pd
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="2"

device = torch.device("cuda:0")

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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "/export/gaon1/data/jteneggi/data/lisa/datasets"
test_dir = os.path.join(data_dir, "test")
dataset = datasets.ImageFolder(test_dir, transform)
image_names = [os.path.basename(sample[0]) for sample in dataset.samples]

true_positives = []
false_positives = []
false_negatives = []
true_negatives = []
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)
for batch_id, (images, labels) in tqdm(enumerate(dataloader)):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    np_labels = labels.cpu().numpy()
    np_preds = preds.detach().cpu().numpy()
    # print(np_labels, np_preds)
    
    for i, label in enumerate(np_labels):
        image_id = batch_id * 4 + i
        image_name = image_names[image_id]
        image = os.path.join(test_dir, str(label), image_name)
        if label == 0:
            if np_preds[i] == 0:
                true_negatives.append(image)
            if np_preds[i] == 1:
                false_positives.append(image)
        if label == 1:
            if np_preds[i] == 1:
                true_positives.append(image)
            if np_preds[i] == 0:
                false_negatives.append(image)
                
correct = 0
wrong = 0
correct += len(true_positives)
correct += len(true_negatives)
wrong += len(false_negatives)
wrong += len(false_positives)
accuracy = correct/(correct+wrong)
print("Test accuracy: %.4f" % accuracy)
np.save("true_positives", true_positives, allow_pickle=True)
print("Saved true positives")