import torch

def datasetMeanStd(loader):
    # Computes the mean and standard deviation of a DataLoader of 3 channel images
    mean = 0.
    std = 0.
    N = len(loader.dataset)
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= N
    std /= (N-1)
    return mean, std