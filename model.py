import time
import torch
from utils import *
import torch.nn as nn
from tqdm import tqdm
from data import Dataset
import torch.optim as optim
from models import vgg, resnet


class model:
    def __init__(self, args):
        self.model = args.model
        self.phase = args.phase
        self.dataroot = args.dataroot
        self.channels = args.c
        self.classes = args.classes

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr

        self.device = args.device
        self.save_freq = args.save_freq
        self.save_path = args.save_path
        self.load_path = args.load_path
        self.image_path = args.image_path
        self.save_type = args.save_type
        self.stop = args.stop

        # Summary information
        print_info(args)

    def build_model(self):
        """
        This function build dataset, model and initialize parameters.
        """
        if self.model == "VGG16":
            self.net = vgg.VGG16(self.channels, self.classes).to(self.device)
        elif self.model == "VGG19":
            self.net = vgg.VGG19(self.channels, self.classes).to(self.device)
        elif self.model == "RESNET18":
            self.net = resnet.RESNET18(self.channels, self.classes).to(self.device)
        elif self.model == "RESNET34":
            self.net = resnet.RESNET34(self.channels, self.classes).to(self.device)
        elif self.model == "RESNET50":
            self.net = resnet.RESNET50(self.channels, self.classes).to(self.device)
        elif self.model == "RESNET152":
            self.net = resnet.RESNET152(self.channels, self.classes).to(self.device)

        # Define dataset
        if self.image_path is None:
            self.data = Dataset(self.dataroot)
            self.trainloader = self.data.loader(self.data.train, self.batch_size)
            self.testloader = self.data.loader(self.data.test, self.batch_size)

        # Weight initialization
        if self.load_path is None:
            print("Weight Initialization")
            init_weight(self.net)
        else:
            print(f"Loading weight from {self.load_path}")
            self.load(self.load_path)
        print()

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def save(self, epoch, save_type):
        path = f"{self.save_path }/{self.model}/"
        if save_type == "N_epoch":
            torch.save(self.net.state_dict(), check_folder(path) + f"{self.model}_{epoch+1}.pth")
        else:
            torch.save(self.net.state_dict(), check_folder(path) + f"{self.model}_best.pth")

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), self.lr, momentum=0.9, weight_decay=1e-3)
        early_stop = early_stopping(self.stop)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        self.net.train()

        iteration = 0
        for epoch in range(self.epoch):
            losses = []
            acc = []
            loop = tqdm((self.trainloader), total=len(self.trainloader), leave=False)
            for i, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()

                optimizer.step()
                scheduler.step(epoch + i / len(self.trainloader))

                correct = 0
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == labels).sum().item()
                accurary = correct / labels.size(0)
                acc.append(accurary)

                iteration += 1
                loop.set_description(f"Epoch [{epoch+1}/{self.epoch}]")
                loop.set_postfix(loss=loss.item(), acc_train=accurary * 100)
                time.sleep(0.1)

            mean_loss = sum(losses) / len(losses)
            mean_acc = sum(acc) / len(acc)
            scheduler.step()
            print(f"Epoch [{epoch+1}/{self.epoch}] Iter: {iteration}  Loss: {mean_loss} Acc: {mean_acc}")

            # Early stopping
            early_stop(mean_loss)
            if early_stop.stop:
                # Save best epoch
                self.save(epoch - self.stop, self.save_type)
                return

            # Save epoch
            if epoch % self.save_freq - 1 == 0:
                self.save(epoch, self.save_type)
            elif (self.save_type == "best_epoch") and (early_stop.count == 0):
                self.save(epoch, self.save_type)

    def test(self):
        # Load model
        self.load(self.load_path)
        self.net.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)

                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Accuracy of the network on the {len(self.data.test)} test images: {100*(correct/total)}%")
