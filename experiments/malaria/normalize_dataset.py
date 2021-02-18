import numpy as np
import torch
from torchvision import datasets, models, transforms
import os
from PIL import Image

DATA_DIR = "/export/gaon1/data/jteneggi/data/malaria/"
IMG_DIR = os.path.join(DATA_DIR, "cropped_images")

from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, rootdir):
        self.rootdir = rootdir
        images = os.listdir(rootdir)
        self.images = images
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.rootdir, self.images[idx]))
        return self.transform(image)

    def __len__(self):
        return len(self.images)


dataset = CustomDataset(IMG_DIR)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=4)
running_count = 0
running_mean = torch.zeros(3)
running_var = torch.zeros(3)
# total_tensor = torch.zeros((1328, 3, 1200*1600))
for i, batch in enumerate(dataloader):
    batch_images = batch.size(0)
    batch_view = batch.view(3, 10, -1).view(3, -1)
    batch_mean = batch_view.mean(1)
    batch_var = batch_view.var(1)
    running_mean = (
        running_count / (running_count + batch_images) * running_mean
        + batch_images / (running_count + batch_images) * batch_mean
    )
    running_var = (
        running_count / (running_count + batch_images) * running_var
        + batch_images / (running_count + batch_images) * batch_var
        + running_count
        * batch_images
        / (running_count + batch_images) ** 2
        * (running_mean - batch_mean) ** 2
    )
    running_count += batch_images
    print(i, batch_images, running_count)
    # batch_view = batch.view(batch_images, batch.size(1), -1)
    # total_tensor[i*10:i*10+10, :, :] = batch_view
    # mean += batch_view.mean(2).sum(0)
    # var += batch_view.var(2).sum(0)
    # imagecount += batch_images

mean = running_mean
print(mean)
std = torch.sqrt(running_var)
print(std)
