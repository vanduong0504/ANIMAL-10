import argparse
from utils import *


class Options:
    """
    This class provide some basic arguments.
    """
    def initialize(self, parser):
        parser.add_argument("--model", type=str, required=True,
                        help="[VGG16, VGG19, RESNET18, RESNET34, RESNET50, RESNET101,...]")
        parser.add_argument("--phase", type=str, default=None,
                            help="[train, test]")
        parser.add_argument("--dataroot", type=str, default=None,
                            help="Path to datasets.")
        parser.add_argument("--channels", default=3, type=int,
                            help="Number of image channels. (default: 3)")
        parser.add_argument("--classes", default=10, type=int,
                            help="Number of classes. (default: 10)")
        parser.add_argument("--epoch", default=20, type=int,
                            help="Number of total epochs. (default: 20)")
        parser.add_argument("--batch_size", default=128, type=int, metavar="BS",
                            help="Input batch size. (default: 128)")
        parser.add_argument("--lr", type=float, default=1e-1,
                            help="Learning rate. (default:1e-1)")
        parser.add_argument("--device", type=str, default="cuda",
                            help="Set gpu mode; [cpu, cuda]")
        parser.add_argument("--save_type", type=str, default='best_epoch', metavar="ST",
                            help=" [best_epoch, N_epochs]")
        parser.add_argument("--save_freq", type=int, default=10, metavar="SF",
                            help="Number of epochs to save latest results.(default: 10)")
        parser.add_argument("--save_path", type=str, default="./weight", metavar="SP",
                            help="Save weight path. (default: `./weight`)")
        parser.add_argument("--load_path", type=str, default=None, metavar="LP",
                            help="Load weight path.")
        parser.add_argument("--image_path", type=str, default=None, metavar="IP",
                            help="Image path for testing single image.")
        parser.add_argument("--stop", type=int, default=99,
                            help="Early stopping. (default:99)")
        return parser

    def check_args(self, args):
        """
        This function to check for the arguments.
        """

        #--epoch
        try:
            assert args.epoch >= 1
        except BaseException:
            print("Number of epoch must be larger than or equal to one")

        #--batch_size
        try:
            assert args.batch_size >= 1
        except BaseException:
            print("batch size must be larger than or equal to one")
        return args

    def gather_options(self):
        parser = argparse.ArgumentParser(description="Image classification models")
        self.parser = self.initialize(parser)

        return self.check_args(self.parser.parse_args())

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        return self.opt
