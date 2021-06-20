import torch
import numpy as np
from utils import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class CIFAR:
    def __init__(self, save_floder='./data'):
        self.floder = save_floder
        self.train = datasets.CIFAR10(root=self.floder, train=True, download=True, transform=self.transform(1))
        self.test = datasets.CIFAR10(root=self.floder, train=False, download=True, transform=self.transform(0))
        self.classes = self.train.class_to_idx

    def transform(self, type):
        # bool type for train or test
        if type:
            return transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            return transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

    def loader(self, dataset, batch_sizes):
        return DataLoader(dataset=dataset, batch_size=batch_sizes, shuffle=True, num_workers=2)
