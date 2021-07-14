from network import Model
from option import Options


def main():
    opt = Options().parse()
    print(opt)
    net = Model(opt)
    net.build_model()

    if net.opt.phase == "train":
        print("[*] Training begin!")
        net.train()
        print("[*] Training finished!")

    if net.opt.phase == "test":
        print("[*] Testing begin!")
        net.test()
        print("[*] Testing finished!")


if __name__ == "__main__":
    main()
