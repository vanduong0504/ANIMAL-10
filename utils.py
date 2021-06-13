import os
import torch
import numpy as np
import torch.nn as nn
from matplotlib import image
from torchvision import utils
import matplotlib.pyplot as plt
from torch.nn.modules import batchnorm
from torch.nn.modules.linear import Linear
from torch.nn.modules.batchnorm import BatchNorm2d

"Mean and std for CIFAR10 after calculate with mean_std function"
mean_train, std_train = (0.4915, 0.4822, 0.4466), (0.2465, 0.2430, 0.2609)
mean_test, std_test = (0.4942, 0.4851, 0.4504), (0.2462, 0.2425, 0.2609)


def mean_std(loader):
    total_means, total_variations, nums_batches = 0, 0, 0
    for data, _ in loader:
        mean_batch, variation = 0, 0
        # mean
        mean_batch = torch.mean(data, dim=[0, 2, 3])

        # variation
        square_batch = torch.pow(data - mean_batch.view(1, -1, 1, 1), 2)
        variation = torch.mean(square_batch, dim=[0, 2, 3])

        total_means += mean_batch
        total_variations += variation
        nums_batches += 1

    means = total_means / nums_batches
    variations = (total_variations / nums_batches)**0.5
    return means, variations


def show_image(loader, batch_size, Type=None):
    "Type: None for Image, True for Train Loader, False for Test Loader"
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(loader)
    images, labels = dataiter.next()
    if Type is True:
        "Train Loader"
        images = reverse_Normalize(images, mean_train, std_train)
    elif Type is False:
        "Test Loader"
        images = reverse_Normalize(images, mean_test, std_test)
    else:
        "Image"
        images = reverse_Normalize(images, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    images = (images * 255).type(torch.uint8)
    grid = utils.make_grid(images)

    # show images
    npimg = grid.numpy()
    plt.imshow(np.transpose(npimg, axes=(1, 2, 0)))
    plt.show()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def Normalize(image, mean, std):
    "Normalize numpy image to tensor range 0-1"
    image = torch.from_numpy(image / 255).permute(2, 0, 1)
    return (image - mean) / std


def reverse_Normalize(x, mean, std):
    n_dim = len(x.size())
    mean = torch.from_numpy(np.array(mean))
    std = torch.from_numpy(np.array(std))

    if n_dim == 4:
        return x * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    else:
        return x * std + mean


def init_weight(net):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.constant_(layer.bias, 0)


class early_stopping():
    def __init__(self, patience):
        self.patience = patience
        self.count = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss > loss:
            self.count = 0
            self.best_loss = loss
        else:
            self.count += 1
            print(f"INFO: Early stopping counter {self.count} of {self.patience}")
            if self.count == self.patience:
                print('INFO: Early stopping')
                self.stop = True
