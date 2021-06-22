import torch
import torch.nn as nn
import collections as cl

# Define dictionary where (key,values) = (blockname,[in_channels, out_channels, repeat])"
Resnet_block = {

    'resnet18': cl.OrderedDict({
        'block_2': [64, 64, 2],
        'block_3': [64, 128, 2],
        'block_4': [128, 256, 2],
        'block_5': [256, 512, 2]}),

    'resnet34': cl.OrderedDict({
        'block_2': [64, 64, 3],
        'block_3': [64, 128, 4],
        'block_4': [128, 256, 6],
        'block_5': [256, 512, 3]}),

    'resnet50': cl.OrderedDict({
        'block_2': [64, 64, 3],
        'block_3': [64, 128, 4],
        'block_4': [128, 256, 6],
        'block_5': [256, 512, 3]}),

    'resnet152': cl.OrderedDict({
        'block_2': [64, 64, 3],
        'block_3': [64, 128, 8],
        'block_4': [128, 256, 36],
        'block_5': [256, 512, 3]})
}


class BasicBlock(nn.Module):
    """
    This class init the residual block with skip connection = 2 for Resnet18 and Resnet34
    """

    def __init__(self, in_cha, out_cha, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_cha)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_cha, out_channels=out_cha, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_cha)
        self.downsample = downsample

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        # Downsample
        if self.downsample is not None:
            output += self.downsample(input)
        else:
            output += input

        output = self.relu(output)
        return output


class Bottleneck(nn.Module):
    """
    This class init the residual block with skip connection = 3 for Resnet50 and Resnet152
    """

    def __init__(self, in_cha, out_cha, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_cha)
        self.conv2 = nn.Conv2d(in_channels=out_cha, out_channels=out_cha, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_cha)
        self.conv3 = nn.Conv2d(in_channels=out_cha, out_channels=out_cha * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_cha * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        # Downsample
        if self.downsample is not None:
            output += self.downsample(input)
        else:
            output += input

        output = self.relu(output)
        return output


class Resnet(nn.Module):

    def __init__(self, resnet_dic, image_channels, nums_class, shortcut):
        super().__init__()
        self.image_channels = image_channels
        self.nums_class = nums_class

        # Coding format follow torchvision.models.restnet
        self.conv1 = nn.Conv2d(in_channels=self.image_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if shortcut == 2:
            self.layer1 = self.layer_block(resnet_dic['block_2'], shortcut, downsample=None)
        elif shortcut == 3:
            self.layer1 = self.layer_block(resnet_dic['block_2'], shortcut, downsample=True)

        self.layer2 = self.layer_block(resnet_dic['block_3'], shortcut, downsample=True, intermediate_channels=True)
        self.layer3 = self.layer_block(resnet_dic['block_4'], shortcut, downsample=True, intermediate_channels=True)
        self.layer4 = self.layer_block(resnet_dic['block_5'], shortcut, downsample=True, intermediate_channels=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        out_flatten = resnet_dic['block_5'][1]
        if shortcut == 2:
            self.classifier = nn.Linear(out_flatten, self.nums_class)
        elif shortcut == 3:
            self.classifier = nn.Linear(out_flatten * 4, self.nums_class)

    def forward(self, input):
        output = self.relu(self.bn1(self.conv1(input)))
        output = self.maxpool1(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.classifier(output)

        return output

    def layer_block(self, resnet_dic, shortcut, downsample=None, intermediate_channels=None):
        block = []
        in_, out_, repeat = resnet_dic

        if downsample is not None:
            if shortcut == 2:
                downsample_block = nn.Sequential(nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=1, stride=2, bias=False),
                                                 nn.BatchNorm2d(num_features=out_))
                block += [BasicBlock(in_cha=in_, out_cha=out_, stride=2, downsample=downsample_block)]

            elif shortcut == 3:
                if intermediate_channels is None:
                    downsample_block = nn.Sequential(nn.Conv2d(in_channels=in_, out_channels=out_ * 4, kernel_size=1, stride=1, bias=False),
                                                     nn.BatchNorm2d(num_features=out_ * 4))
                    block += [Bottleneck(in_cha=in_, out_cha=out_, stride=1, downsample=downsample_block)]
                else:
                    downsample_block = nn.Sequential(nn.Conv2d(in_channels=in_ * 4, out_channels=out_ * 4, kernel_size=1, stride=1, bias=False),
                                                     nn.BatchNorm2d(num_features=out_ * 4))
                    block += [Bottleneck(in_cha=in_ * 4, out_cha=out_, stride=1, downsample=downsample_block)]
        else:
            if shortcut == 2:
                block += [BasicBlock(in_cha=in_, out_cha=out_, stride=1)]

            elif shortcut == 3:
                block += [Bottleneck(in_cha=in_, out_cha=out_, stride=1)]

        for i in range(1, repeat):
            if shortcut == 2:
                block += [BasicBlock(in_cha=out_, out_cha=out_, stride=1)]

            elif shortcut == 3:
                block += [Bottleneck(in_cha=out_ * 4, out_cha=out_, stride=1)]

        return nn.Sequential(*block)


def RESNET18(img_channel=3, num_classes=10):
    return Resnet(Resnet_block['resnet18'], img_channel, num_classes, shortcut=2)


def RESNET34(img_channel=3, num_classes=10):
    return Resnet(Resnet_block['resnet34'], img_channel, num_classes, shortcut=2)


def RESNET50(img_channel=3, num_classes=10):
    return Resnet(Resnet_block['resnet50'], img_channel, num_classes, shortcut=3)


def RESNET152(img_channel=3, num_classes=10):
    return Resnet(Resnet_block['resnet152'], img_channel, num_classes, shortcut=3)
