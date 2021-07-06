import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import utils
import matplotlib.pyplot as plt

# Mean and std of ImageNet
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def mean_std(loader):
    """
    This function use to calculate mean and standard deviation of your dataset. 
    This mean and std use in data.py to normalze your data.
    """
    total_means, total_variations, nums_batches = 0, 0, 0
    for data, _ in loader:
        mean_batch, variation = 0, 0
        # Mean
        mean_batch = torch.mean(data, dim=[0, 2, 3])

        # Variation
        square_batch = torch.pow(data - mean_batch.view(1, -1, 1, 1), 2)
        variation = torch.mean(square_batch, dim=[0, 2, 3])

        total_means += mean_batch
        total_variations += variation
        nums_batches += 1

    means = total_means / nums_batches
    variations = (total_variations / nums_batches)**0.5
    return means, variations


def show_image(classes=None, image_path=None, loader=None):
    """
    This function use to show_image for batch from loader or single image.
    """
    if image_path is not None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        grid = utils.make_grid(image)
    else:
        dataiter = iter(loader)
        batch_images, labels = dataiter.next()
        batch_images = reverse_Normalize(batch_images, mean, std)
        batch_images = (batch_images * 255).type(torch.uint8).permute(1, 2, 0)
        grid = utils.make_grid(batch_images)
        print(' '.join(classes[label] for label in labels))

    plt.imshow(grid.numpy())
    plt.show()
    return grid


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


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

        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.constant_(layer.bias, 0)


def print_info(opt):
    print("##### Information #####")
    print("# model : ", opt.model)
    print("# channels : ", opt.channels)
    if opt.phase == "train":
        print("# classes : ", opt.classes)
        print("# epoch : ", opt.epoch)
        print("# learning rate : ", opt.lr)
        print("# batch_size : ", opt.batch_size)
        print("# save_freq  : ", opt.save_freq)
    elif opt.phase == "test":
        print("# classes : ", opt.classes)
    else:
        print("# weight path : ", opt.load_path)
        print("# image path : ", opt.image_path)
    print("#######################", end='\n\n')


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