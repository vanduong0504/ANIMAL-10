import time
import torch
from utils import *
import torch.nn as nn
from tqdm import tqdm
from data import Dataset
import torch.optim as optim
from models import vgg, resnet


class model:
    def __init__(self, opt):
        self.opt = opt

        # Summary information
        print_info(self.opt)

    def build_model(self):
        """
        This function build dataset, model and initialize parameters.
        """
        name = self.opt.model
        if name.find("VGG"):
            self.net = vgg.create_model(name, self.opt.c, self.opt.classes).to(self.opt.device)
        elif name.find("RESNET"):
            self.net = resnet.create_model(name, self.opt.c, self.opt.classes).to(self.opt.device)

        # Define dataset
        if self.opt.image_path is None:
            self.data = Dataset(self.opt.dataroot)
            self.trainloader = self.data.loader(self.data.train, self.opt.batch_size)
            self.testloader = self.data.loader(self.data.test, self.opt.batch_size)

        # Weight initialization
        if self.opt.load_path is None:
            print("Weight Initialization")
            init_weight(self.net)
        else:
            print(f"Loading weight from {self.opt.load_path}")
            self.load(self.opt.load_path)
        print()

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def save(self, epoch, save_type):
        path = f"{self.opt.save_path }/{self.model}/"
        if save_type == "N_epoch":
            torch.save(self.net.state_dict(), check_folder(path) + f"{self.model}_{epoch+1}.pth")
        else:
            torch.save(self.net.state_dict(), check_folder(path) + f"{self.model}_best.pth")

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), self.opt.lr, momentum=0.9, weight_decay=1e-3)
        early_stop = early_stopping(self.opt.stop)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        self.net.train()

        iteration = 0
        for epoch in range(self.opt.epoch):
            losses = []
            acc = []
            loop = tqdm((self.trainloader), total=len(self.trainloader), leave=False)
            for i, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(self.opt.device), labels.to(self.opt.device)
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
                loop.set_description(f"Epoch [{epoch+1}/{self.opt.epoch}]")
                loop.set_postfix(Loss=loss.item(), Acc_train=accurary * 100)
                time.sleep(0.1)

            mean_loss = sum(losses) / len(losses)
            mean_acc = sum(acc) / len(acc)
            scheduler.step()
            print(f"Epoch [{epoch+1}/{self.opt.epoch}] Iter: {iteration} Acc: {mean_acc} Loss: {mean_loss}")

            # Early stopping
            early_stop(mean_loss)
            if early_stop.stop:
                # Save best epoch
                self.save(epoch - self.opt.stop, self.opt.save_type)
                return

            # Save epoch
            if epoch % self.save_freq - 1 == 0:
                self.save(epoch, self.opt.save_type)
            elif (self.opt.save_type == "best_epoch") and (early_stop.count == 0):
                self.save(epoch, self.opt.save_type)

    def test(self):
        # Load model
        self.load(self.opt.load_path)
        self.net.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in self.testloader:
                inputs, labels = inputs.to(self.opt.device), labels.to(self.opt.device)
                outputs = self.net(inputs)

                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Accuracy of the network on the {len(self.data.test)} test images: {100*(correct/total)}%")
