import os
from utils import mean, std
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Dataset:
    def __init__(self, dataroot):
        self.train = datasets.ImageFolder(root=os.path.join(dataroot, 'train'), transform=self.transform(1))
        self.test = datasets.ImageFolder(root=os.path.join(dataroot, 'test'), transform=self.transform(0))
        self.classes = self.train.class_to_idx

    @staticmethod
    def transform(type):
        # Bool type for train or test
        if type:
            return transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(224, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            return transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

    @staticmethod
    def loader(dataset, batch_sizes):
        return DataLoader(dataset=dataset, batch_size=batch_sizes, shuffle=True, num_workers=2)
