import torch
import torch.nn as nn

# Define dictionary where (key,values) = (blockname: [nums_conv2d,in_channels,out_channels])"
VGG_block = {

    'vgg16': {
        'block_1': [2, 'image_channels', 64],
        'block_2': [2, 64, 128],
        'block_3': [3, 128, 256],
        'block_4': [3, 256, 512],
        'block_5': [3, 512, 512]},

    'vgg19': {
        'block_1': [2, 'image_channels', 64],
        'block_2': [2, 64, 128],
        'block_3': [4, 128, 256],
        'block_4': [4, 256, 512],
        'block_5': [4, 512, 512]}
}

# Define dictionary where (key,values) = (fc: [in_features,out_features])"
fc = {'fc1': [25088, 4096], 'fc2': [4096, 4096], 'fc3': [4096, 'num_class']}


class VGG(nn.Module):
    def __init__(self, vgg_dic, image_channels, nums_class):
        super().__init__()
        self.image_channels = image_channels
        self.nums_class = nums_class

        features = []
        for values in vgg_dic.values():
            features += self.feature_block(*values)

        # Coding format torchvision.models.vgg
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=7)
        self.classifier = nn.Sequential(*self.classifier_block(fc))

    def forward(self, input):
        output = self.features(input)
        output = self.avgpool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.classifier(output)

        return output

    def feature_block(self, nums_conv, in_cha, out_cha):
        block = []
        for _ in range(nums_conv):
            try:
                block += [nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=3, padding=1, bias=False)]
            except TypeError:
                block += [nn.Conv2d(in_channels=self.image_channels, out_channels=out_cha, kernel_size=3, padding=1, bias=False)]
            block += [nn.BatchNorm2d(num_features=out_cha)]
            block += [nn.ReLU(inplace=True)]
            in_cha = out_cha
        block += [nn.MaxPool2d(kernel_size=2)]
        return block

    def classifier_block(self, fc):
        block = []
        for values in fc.values():
            in_fea, out_fea = values
            if isinstance(out_fea, int):
                block += [nn.Linear(in_features=in_fea, out_features=out_fea)]
                block += [nn.ReLU(inplace=True)]
                block += [nn.Dropout()]
            else:
                block += [nn.Linear(in_features=in_fea, out_features=self.nums_class)]
        return block


def create_model(name, img_channel, num_classes):
    if name.find('16'):
        return VGG(VGG_block['vgg16'], img_channel, num_classes)
    else:
        return VGG(VGG_block['vgg19'], img_channel, num_classes)

