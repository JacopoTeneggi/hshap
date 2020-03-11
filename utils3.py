import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def datasetMeanStd(loader):
    # Computes the mean and standard deviation of a DataLoader of 3 channel images
    mean = 0.
    std = 0.
    N = len(loader.dataset)
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= N
    std /= (N-1)
    return mean.numpy(), std.numpy()


def denormalize(im, mean, std):
  return im*std + mean

def input2image(input, mean, std):
    sample_image = input.numpy().transpose(1, 2, 0)
    return denormalize(sample_image, mean, std)

def display_image(im, true_label, predicted_label=None, figure_size = (12, 10)):
    plt.figure(figsize=figure_size)
    plt.imshow(im)
    title_ = "True label : " + str(true_label)
    if (predicted_label != None):
        title_ += "/ Predicted : " + str(predicted_label)
    plt.title(title_)

def almost_equal(n1, n2, e):
  return abs(n1-n2) < e

def network_has_converged(loss, e):
  if (len(loss) < 3):
    return False
  else:
    return (almost_equal(loss[-3], loss[-2], e) and
            almost_equal(loss[-3], loss[-1], e) )

def training_accuracy(network, loader):
  with torch.no_grad():
    correct = 0
    total = 0
    for data in loader:
      images, labels = data
      outputs = network(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return 100 * correct/total

def validation_stats(network, loader, criterion):
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

  return (100 * correct/total, total_loss/len(loader.dataset))

def plot_training(train_loss, val_loss, train_accuracy, val_accuracy):
  fig, (ax1, ax2) = plt.subplots(1,3, figsize = (15,5))
  x_scale = np.linspace(0, len(train_loss)- 1, len(train_loss) )
  _ = ax1.plot(x_scale, train_loss)
  _ = ax1.plot(x_scale, val_loss)
  ax1.legend(["Loss on the training set", "Loss on the validation set"])
  ax1.xlabel("Number of generations")
  ax1.ylabel("Evaluation of the loss function")

  x_scale = np.linspace(0, len(train_accuracy)- 1, len(train_accuracy) )
  _ = ax2.plot(x_scale, train_accuracy)
  _ = ax2.plot(x_scale, val_accuracy)
  ax2.legend(["Training accuracy", "Validation accuracy"])
  ax2.xlabel("Number of generations")
  ax2.ylabel("Accuracy ")


def test(net, loader):
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
            for truth in (predicted == labels):
                if not truth:
                    wrong_im.append(images[k])
                    wrong_label.append(int(labels[k]))
                    wrongly_predicted_label.append(int(predicted[k]))
                k += 1

    print("Accuracy of the network on the " + str(total) + ' test images: %.3f %%' % (
            100 * correct / total))

    print("Number of mistakes : " + str(total - correct))
    return wrong_im, wrong_label, wrongly_predicted_label


def train(net, optimizer, criterion, max_epochs, dataloader, valloader, , epsilon):
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

            print('Generation %d. training loss: %.4f,'
                  % (epoch + 1, train_loss[-1]), end="")
            print(" training accuracy: %.2f " % (train_accuracy[-1]), end="%,")
            print(" validation loss: %.4f," % (val_loss[-1]), end=" ")
            print(" validation accuracy: %.2f " % (val_accuracy[-1]), end="% \n")

            converged = network_has_converged(train_loss, epsilon)

    if (converged):
        print("Network has converged.")
    else:
        print("Network hasn't been able to converge in " + str(max_epochs) + " generations.")
    return train_loss, val_loss, train_accuracy, val_accuracy

class Net(nn.Module):

    def __init__(self):
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
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

