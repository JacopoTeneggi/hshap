# PYTORCH IMPORT
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# PIL IMPORT
from PIL import Image

# MATPLOTLIB IMPORT
import matplotlib.pyplot as plt

# SYS IMPORTS
import os
import glob


class RSNASickDataset(data.Dataset):
    """Custom Sick RSNA Dataset"""

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.samples = glob.glob(os.path.join(root_dir, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample

    def loader(self, path):
        from torchvision import get_image_backend

        if get_image_backend() == "accimage":
            import accimage

            try:
                return accimage.Image(path)
            except IOError:
                # Potentially a decoding problem, fall back to PIL.Image
                return pil_loader(path)
        else:
            return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def datasetMeanStd(loader):
    """ Computes the mean and standard deviation of a dataloader of 3 channel images

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        the dataloader whose mean is computed

    Returns
    -------
    mean : numpy array of shape (3,)
        the channel-wise mean of the dataloader
    std : numpy array of shape (3,)
        the channel-wise standard deviation of the dataloader
    """

    mean = 0.0
    std = 0.0
    N = len(loader.dataset)
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= N
    std /= N - 1
    mean = mean.numpy()
    std = std.numpy()
    return mean, std


def denormalize(im, mean, std):
    """ Restore an image to its range before normalization

    Parameters
    ----------
    im : numpy array with 3 channels
        the image to denormalize
    mean : numpy array of shape (3,)
        the channel-wise mean of the dataloader
    std : numpy array of shape (3,)
        the channel-wise standard deviation of the dataloader

    Returns
    -------
    denorm : numpy array with 3 channels
        the denormalized image
    """

    denorm = im * std + mean
    return denorm


def input2image(input, mean, std):
    """ Convert an torch tensor input to a neural net as a image in numpy array format, denormalized.

    Parameters
    ----------
    input : torch tensor with 3 channels
        the input tensor to process
    mean : numpy array of shape (3,)
        the channel-wise mean of the dataloader
    std : numpy array of shape (3,)
        the channel-wise standard deviation of the dataloader

    Returns
    -------
    denorm : numpy array with 3 channels
        the denormalized image
    """

    sample_image = input.numpy().transpose(1, 2, 0)
    denorm = denormalize(sample_image, mean, std)
    return denorm


def display_image(im, true_label, predicted_label=None, figure_size=(8, 5)):
    """ Convert an torch tensor input to a neural net as a image in numpy array format, denormalized.

    Parameters
    ----------
    im : numpy array with 3 channels
        the image to display
    true_label : int in {0,1}
        the ground truth label of im
    true_label : int in {0,1}, optional 
        the predicted truth label of im by the neural network 
    figure_size : tuple of ints, optional 
        size of the matplotlib.pyplot.figure object
    """

    plt.figure(figsize=figure_size)
    plt.imshow(im)
    title_ = "True label : " + str(true_label)
    if predicted_label != None:
        title_ += "/ Predicted : " + str(predicted_label)
    plt.title(title_)


def almost_equal(n1, n2, e):
    """ Determine if n1 and n2 are almost equal, at a tolerance e.

    Parameters
    ----------
    n1 : float
        the first number
    n2 : float
        the second number
    e : float
        the tolerance

    Returns
    -------
    verdict : bool
        whether or not they are almost equal
    """

    verdict = abs(n1 - n2) < e
    return verdict


def network_has_converged(loss, e):
    """ Determine if n1 and n2 are almost equal, at a tolerance e.

    Parameters
    ----------
    loss : list of floats
        the list of losses during training
    e : float
        the tolerance for the stopping criterion

    Returns
    -------
    converged : bool
        whether or not training has converged
    """

    if len(loss) < 3:
        converged = False
    else:
        converged = almost_equal(loss[-3], loss[-2], e) and almost_equal(
            loss[-3], loss[-1], e
        )
    return converged


def training_accuracy(network, loader):
    """ Determine the accuracy of the predictions of a network.

    Parameters
    ----------
    network : torch.nn.Module or subclass
        the network to test
    loader : torch.utils.data.DataLoader
        the training dataloader

    Returns
    -------
    accuracy : float betwee 0. and 100.
        the training accuracy
    """

    with torch.no_grad():
        correct = 0
        total = 0
        for data in loader:
            images, labels = data
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def validation_stats(network, loader, criterion):
    """ Judge the training of the network on the validation set.

    Parameters
    ----------
    network : torch.nn.Module or subclass
        the network to test
    loader : torch.utils.data.DataLoader
        the validation dataloader
    criterion : torch.nn loss function
        the loss function criterion

    Returns
    -------
    accuracy : float between 0. and 100.
        the validation accuracy
    normalized_loss : float
        the loss per input
    """

    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    accuracy, normalized_loss = (
        100 * correct / total,
        total_loss / len(loader.dataset),
    )
    return accuracy, normalized_loss


def plot_training(train_loss, val_loss, train_accuracy, val_accuracy):
    """ Plot the training loss and accuracy of the network on the training and validation set.

    Parameters
    ----------
    train_loss : list of floats
        the loss on the training set
    val_loss : list of floats
        the loss on the validation set
    train_accuracy : float between 0. and 100.
        the accuracy on the training set
    val_accuracy : float between 0. and 100.
        the accuracy on the validations set
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    x_scale = np.linspace(0, len(train_loss) - 1, len(train_loss))
    _ = ax1.plot(x_scale, train_loss)
    _ = ax1.plot(x_scale, val_loss)
    ax1.legend(["Loss on the training set", "Loss on the validation set"])
    ax1.set_xlabel("Number of generations")
    ax1.set_ylabel("Evaluation of the loss function")

    x_scale = np.linspace(0, len(train_accuracy) - 1, len(train_accuracy))
    _ = ax2.plot(x_scale, train_accuracy)
    _ = ax2.plot(x_scale, val_accuracy)
    ax2.legend(["Training accuracy", "Validation accuracy"])
    ax2.set_xlabel("Number of generations")
    ax2.set_ylabel("Accuracy ")


def test(net, loader):
    """ Confront the network to the testing set.

    Parameters
    ----------
    net : torch.nn.Module or subclass
        the network to test
    loader : torch.utils.data.DataLoader
        the testing dataloader

    Returns
    -------
    wrong_im : list of torch tensors
        inputs classified wrongly
    wrong_label : list of ints ({0,1})
        corresponding labels
    wrong_label : list of ints ({0,1})
        corresponding predictions (redundant)
    """

    correct = 0
    total = 0
    wrong_im = []
    wrong_label = []
    wrongly_predicted_label = []

    with torch.no_grad():
        for data in loader:
            images, labels = data

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            k = 0
            for truth in predicted == labels:
                if not truth:
                    wrong_im.append(images[k])
                    wrong_label.append(int(labels[k]))
                    wrongly_predicted_label.append(int(predicted[k]))
                k += 1

    print(
        "Accuracy of the network on the "
        + str(total)
        + " test images: %.3f %%" % (100 * correct / total)
    )

    print("Number of mistakes : " + str(total - correct))
    return wrong_im, wrong_label, wrongly_predicted_label


def train(net, optimizer, criterion, max_epochs, dataloader, valloader):
    """ Train a CNN.

    Parameters
    ----------
    net : torch.nn.Module or subclass
        the network to test
    optimizer : instance of a torch.optim class
        the optimizer for the CNN
    criterion : torch.nn loss function
        the loss function criterion
    max_epochs : int
        maximum number of epochs before terminating training early
    dataloader : torch.utils.data.DataLoader
        dataloader for the training set
    valloader : torch.utils.data.DataLoader
        dataloader for the validation set


    Returns
    -------
    train_loss : list of floats
        the loss on the training set
    val_loss : list of floats
        the loss on the validation set
    train_accuracy : float between 0. and 100.
        the accuracy on the training set
    val_accuracy : float between 0. and 100.
        the accuracy on the validations set
    """

    converged = False
    epsilon = 0.0001
    train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []
    for epoch in range(max_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        if not converged:
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # plot loss
                running_loss += loss.item()

            train_loss.append(running_loss / len(dataloader.dataset))
            train_accuracy.append(training_accuracy(net, dataloader))
            A, L = validation_stats(net, valloader, criterion)
            val_loss.append(L)
            val_accuracy.append(A)

            print(
                "Generation %d. training loss: %.4f," % (epoch + 1, train_loss[-1]),
                end="",
            )
            print(" training accuracy: %.2f " % (train_accuracy[-1]), end="%,")
            print(" validation loss: %.4f," % (val_loss[-1]), end=" ")
            print(" validation accuracy: %.2f " % (val_accuracy[-1]), end="% \n")

            converged = network_has_converged(train_loss, epsilon)

    if converged:
        print("Network has converged.")
    else:
        print(
            "Network hasn't been able to converge in "
            + str(max_epochs)
            + " generations."
        )
    return train_loss, val_loss, train_accuracy, val_accuracy


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
        x = x.view(-1, self.num_flat_features(x))  # 16*9*11
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


class HdNet(nn.Module):
    """ CNN architecture for classifying images of 1200 pixels in width and 1000 in height.
        """

    def __init__(self):
        """ Initialize the CNN.
        """

        super(HdNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=5)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(6, 10, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 16, 4)
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

        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))  # 16*9*11
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
