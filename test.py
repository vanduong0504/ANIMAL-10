import torch
from utils import *
from model import model
from option import Options


def main():
    """
    This function use to test an image and output a class name
    """
    opt = Options().parse()
    net = model(opt)
    net.build_model()

    # Label dictionary for animal10:
    label = {'butterfly': 0, 'cat': 1, 'chicken': 2, 'cow': 3, 'dog': 4, 
    'elephant': 5, 'horse': 6, 'sheep': 7, 'spider': 8, 'squirrel': 9}

    if net.image_path is not None:
        image = show_image(image_path=net.image_path)
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = (image / 255).to(net.device)

        outputs = net.net(image)
        predicted = torch.argmax(outputs, 1).item()

        print(list(label.keys())[predicted])


if __name__ == "__main__":
    main()