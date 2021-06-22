import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import utils
import matplotlib.pyplot as plt
from torch.nn.modules import batchnorm
from torch.nn.modules.linear import Linear
from torch.nn.modules.batchnorm import BatchNorm2d

# Mean and std for CIFAR10 after calculate with mean_std function
mean, std = (0.4915, 0.4822, 0.4466), (0.2465, 0.2430, 0.2609)


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


def show_image(classes=None, image_path=None, loader=None):
    """This function use to show_image for batch from loader or
    image from folder."""
    if image_path is not None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        grid = utils.make_grid(image)
        print(grid.shape)
    else:
        dataiter = iter(loader)
        batch_images, labels = dataiter.next()
        batch_images = reverse_Normalize(batch_images, mean, std)
        batch_images = (batch_images * 255).type(torch.uint8).permute(1,2,0)
        grid = utils.make_grid(batch_images)
        print(' '.join(classes[label] for label in labels))

    # show images
    #npimg = grid.numpy()
    plt.imshow(grid.numpy())
    plt.show()

    return grid


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def Normalize(image, mean, std):
    image = torch.from_numpy(image / 255).permute(2, 0, 1)
    return (image - mean) / std


def reverse_Normalize(x, mean, std):
    mean = torch.from_numpy(np.array(mean))
    std = torch.from_numpy(np.array(std))

    return x * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


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


def print_info(args):
    print()
    print("##### Information #####")
    print("# model : ", args.model)
    print("# dataset : ", args.dataset)
    print("# channels : ", args.c)
    if args.phase == "train":
        print("# classes : ", args.classes)
        print("# epoch : ", args.epoch)
        print("# batch_size : ", args.batch_size)
        print("# save_freq  : ", args.save_freq)
    elif args.phase == "test":
        print("# classes : ", args.classes)
    else:
        print("# weight path : ", args.load_path)
        print("# image path : ", args.image_path)
    print("#######################")
    print()