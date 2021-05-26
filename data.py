import torch
import numpy as np
from utils import *
from torchvision.datasets import cifar
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class ToTensor:
    def __call__(self, sample):
        image = np.array(sample)
        image = torch.from_numpy(image).type(torch.float32)
        image = torch.Tensor.permute(image, dims = [2, 0, 1])
        return image/255

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.from_numpy(np.array(mean)).type(torch.float32)
        self.std = torch.from_numpy(np.array(std)).type(torch.float32)
    
    def __call__(self, sample):
        return (sample - self.mean.view(-1,1,1)) / self.std.view(-1,1,1)

class CIFAR:
    def __init__(self, save_floder='./Data'):
        self.floder = save_floder
        self.train = datasets.CIFAR10(root = self.floder, train=True, download=True, transform=self.transform(type=1))
        self.test = datasets.CIFAR10(root = self.floder, train=False, download=True, transform=self.transform(type=0))

    def transform(self, type):
        "bool type for train or test"
        if type:
            print('Train Transform')
            return transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                ToTensor(),
                Normalize(mean_train, std_train)])
        else:
            print('Test Transform')
            return transforms.Compose(
                [ToTensor(),
                Normalize(mean_test, std_test)])

    def loader(self, dataset, batch_sizes):
        return  DataLoader(dataset = dataset, batch_size = batch_sizes, shuffle = True, num_workers=2)


