from models import *
from utils import *
from model import model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR 10 Image classification")
    parser.add_argument("--model", type=str, required=True, 
                        help="[VGG16, VGG19, RESNET18, RESNET34, RESNET50, RESNET101,...]")
    parser.add_argument("--phase", type=str, default="train", 
                        help="[train / test]")
    parser.add_argument("--dataroot", type=str, default="./data",
                        help="Path to datasets. (default: `./data`)")
    parser.add_argument("--dataset", type= str, default="CIFAR10",
                        help="Dataset name.  (default: `CIFAR10`)")
    parser.add_argument("--c", default=3, type=int, metavar="CHANNELS",
                        help="Number of image channels. (default: 3)")     
    parser.add_argument("--classes", default=10, type=int,
                        help="Number of classes. (default: 10)")                                      
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of total epochs to run. (default: 100)")
    parser.add_argument("--batch_size", default=128, type=int, metavar="BS",
                        help="Input batch size. (default: 128)")
    parser.add_argument("--lr", type=float, default=3e-4, 
                        help="Learning rate. (default:3e-4)")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Set gpu mode; [cpu, cuda]")
    parser.add_argument("--result_dir", type=str, default="./results", metavar="RD",
                        help="Directory name to save the results. (default: `./results`)")
    parser.add_argument("--save_type", type=str, default='best_epoch', metavar="ST",
                        help=" [best_epoch / N_epochs]. (default: `best_epoch`)")
    parser.add_argument("--save_freq", type=int, default=10, metavar="SF",
                        help="The number of epochs to save the latest results. (default: 10)")
    parser.add_argument("--save_path", type=str, default="./weight", metavar="SP",
                        help="Path to weight. (default: `./weight`)")
    parser.add_argument("--stop", type=int, default=10,
                        help="Early stopping. (default: 10)")
    return check_args(parser.parse_args())

def check_args(args):
    "--result_dir"
    check_folder(os.path.join(args.result_dir, args.dataset, "img"))
    
    "--epoch"
    try:
        assert args.epoch >= 1
    except:
        print("number of epochs must be larger than or equal to one")

    "--batch_size"
    try:
        assert args.batch_size >= 1
    except:
        print("batch size must be larger than or equal to one")
    return args

def main():
    "parse arguments"
    args = parse_args()
    if args is None:
      exit()

    net = model(args)
    net.build_model()
    
    if args.phase == "train" :
        print("[*] Training begin!")
        net.train()
        print("[*] Training finished!")

    if args.phase == "test" :
        print("[*] Testing begin!")
        net.test()
        print("[*] Test finished!")

if __name__ == "__main__":
    main()


