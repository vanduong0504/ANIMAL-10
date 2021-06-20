import torch
import numpy as np
from utils import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class CIFAR:
    def __init__(self, save_floder='./data'):
        self.floder = save_floder
        self.train = datasets.CIFAR10(root=self.floder, train=True, download=True, transform=self.transform(type=1))
        self.test = datasets.CIFAR10(root=self.floder, train=False, download=True, transform=self.transform(type=0))
        # #self.classes = ('plane', 'car', 'bird', 'cat',
        #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        print(self.train.classes)

    def transform(self, type):
        #bool type for train or test
        if type:
            return transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean_train, std_train)])
        else:
            return transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean_test, std_test)])

    def loader(self, dataset, batch_sizes):
        return DataLoader(dataset=dataset, batch_size=batch_sizes, shuffle=True, num_workers=2)

A = CIFAR()