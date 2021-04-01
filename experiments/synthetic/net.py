import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    """ CNN architecture for classifying images of 120 pixels in width and 100 in height.
    """

    def __init__(self):
        """ Initialize the CNN.
        """

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.pool2 = nn.MaxPool2d(5)
        self.fc1 = nn.Linear(16 * 9 * 11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        """ Perform forward propagation across the network.
        Parameters
        ----------
        x : torch tensor
            CNN input tensor images
        Returns
        -------
        x : torch tensor
            CNN output class labels
        """

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1, self.num_flat_features(x))  # 16*9*11
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """ Compute the number entries of one item in a batch.
        Parameters
        ----------
        x : torch tensor
            batch of inputs or processed inputs
        Returns
        -------
        num_features : int
            number of entries in one item
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features