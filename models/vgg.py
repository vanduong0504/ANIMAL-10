import torch
import torch.nn as nn
import collections as cl

# Define dictionary where (key,values) = (blockname,[nums_conv2d,in_channels,out_channels])"
VGG_block = {

    'vgg16': cl.OrderedDict({
        'block_1': [2, 'image_channels', 64],
        'block_2': [2, 64, 128],
        'block_3': [3, 128, 256],
        'block_4': [3, 256, 512],
        'block_5': [3, 512, 512]}),

    'vgg19': cl.OrderedDict({
        'block_1': [2, 3, 64],
        'block_2': [2, 64, 128],
        'block_3': [4, 128, 256],
        'block_4': [4, 256, 512],
        'block_5': [4, 512, 512]})
}

# Define dictionary where (key,values) = (fc,[in_features,out_features])"
fc = cl.OrderedDict({'fc1': [25088, 4096], 'fc2': [4096, 4096], 'fc3': [4096, 'num_class']})


class VGG(nn.Module):

    def __init__(self, vgg_dic, image_channels, nums_class):
        super().__init__()
        self.image_channels = image_channels
        self.nums_class = nums_class

        features = []
        for values in vgg_dic.values():
            nums_convs, in_channels, out_channels = values
            features += self.feature_block(nums_convs, self.image_channels, in_channels, out_channels)

        "Coding format torchvision.models.vgg"
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(*self.classifier_block(fc, self.nums_class))

    def forward(self, input):
        output = self.features(input)
        output = self.avgpool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.classifier(output)

        return output

    def feature_block(self, nums_conv, image_channels, in_cha, out_cha):
        block = []
        for _ in range(nums_conv):
            try:
                block += [nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=3, padding=1, bias=False)]
            except BaseException:
                block += [nn.Conv2d(in_channels=image_channels, out_channels=out_cha, kernel_size=3, padding=1, bias=False)]
            block += [nn.BatchNorm2d(num_features=out_cha)]
            block += [nn.ReLU(inplace=True)]
            in_cha = out_cha
        block += [nn.MaxPool2d(kernel_size=(2, 2))]
        return block

    def classifier_block(self, fc, num_class):
        block = []
        for _, values in enumerate(fc.values(), 1):
            in_fea, out_fea = values
            try:
                block += [nn.Linear(in_features=in_fea, out_features=out_fea)]
                block += [nn.ReLU(inplace=True)]
                block += [nn.Dropout()]
            except BaseException:
                block += [nn.Linear(in_features=in_fea, out_features=num_class)]
        return block


def VGG16(img_channel=3, num_classes=10):
    return VGG(VGG_block['vgg16'], img_channel, num_classes)


def VGG19(img_channel=3, num_classes=10):
    return VGG(VGG_block['vgg19'], img_channel, num_classes)


from torchinfo import summary

model = VGG19()
summary(model, (1, 3, 224, 224), depth=3)