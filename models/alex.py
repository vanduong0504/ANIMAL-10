import torch
import torch.nn as nn

# Define dictionary where (key,values) = (conv2d_block: [in_channels, out_channels, kernel_size, stride, padding, maxpool])"
ALEX_block = {

    'block_1': ['image_channels', 96, 11, 4, 2, 1],
    'block_2': [96, 256, 5, 1, 2, 1],
    'block_3': [256, 384, 3, 1, 1, 0],
    'block_4': [384, 384, 3, 1, 1, 0],
    'block_5': [384, 256, 3, 1, 1, 1]
}

# Define dictionary where (key,values) = (fc: [in_features,out_features])"
fc = {'fc1': [9216, 4096], 'fc2': [4096, 4096], 'fc3': [4096, 'num_class']}


class AlexNet(nn.Module):
    def __init__(self, image_channels, nums_class):
        super().__init__()
        self.image_channels = image_channels
        self.nums_class = nums_class

        features = []
        for values in ALEX_block.values():
            features += self.feature_block(*values)

        # Coding format torchvision.models.alexnet
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=6)
        self.classifier = nn.Sequential(*self.classifier_block(fc))

    def forward(self, input):
        output = self.features(input)
        output = self.avgpool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.classifier(output)

        return output

    def feature_block(self, *agrs):
        block = []
        try:
            block += [nn.Conv2d(*agrs[0:5])]
        except TypeError:
            block += [nn.Conv2d(self.image_channels, *agrs[1:5])]
        block += [nn.ReLU(inplace=True)]
        if agrs[5]:
            block += [nn.MaxPool2d(kernel_size=3, stride=2)]
        return block

    def classifier_block(self, fc):
        block = []
        for values in fc.values():
            in_fea, out_fea = values
            if isinstance(out_fea, int):
                block += [nn.Dropout()]
                block += [nn.Linear(in_features=in_fea, out_features=out_fea)]
                block += [nn.ReLU(inplace=True)]
            else:
                block += [nn.Linear(in_features=in_fea, out_features=self.nums_class)]
        return block

def create_model(img_channel, num_classes):
    print("AlexNet created")
    return AlexNet(img_channel, num_classes)
