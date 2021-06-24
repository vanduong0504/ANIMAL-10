import torch
from utils import *
from model import model
from main import parse_args


def main():
    """
    This function use to test an image and output a class name
    """
    net = model(parse_args())
    net.build_model()

    # This is label dictionary for cifar10:
    label = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3,
             'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    if net.image_path is not None:
        image = show_image(image_path=net.image_path)
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = (image / 255).to(net.device)

        outputs = net.net(image)
        predicted = torch.argmax(outputs, 1).item()

        print(list(label.keys())[predicted])


if __name__ == "__main__":
    main()
