import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os

HOME = "/home/jacopo"

# LOAD TRAIN DATA
# USE IMAGE NET MEANS AND STDS
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.299, 0.224, 0.225])

train_data_dir = os.path.join(HOME, "repo/hshap/data/rsna/datasets/train")
preprocess = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
)
train_batch_size = 100
train_data = datasets.ImageFolder(train_data_dir, transform=preprocess)
dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=train_batch_size, shuffle=False, num_workers=10, pin_memory=True
)
train_loader = iter(dataloader)
train_sum = torch.zeros((3, 299, 299))
train_sum = train_sum.to('cuda:0')
batch_count = 0
for batch in train_loader:
    X, _ = batch
    X = X.to('cuda:0')
    batch_mean = torch.mean(X, dim=0)
    train_sum += batch_mean
    batch_count += 1
    print(batch_count)
mean = train_sum / batch_count
mean_npy = mean.cpu().numpy()
plt.imshow()
OUTPUT_PATH = os.path.join(HOME, "repo/hshap/data/rsna/references/avg_all_train.npy")
np.save(OUTPUT_PATH, mean_npy)