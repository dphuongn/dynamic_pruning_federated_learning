import os

import torch
import torchvision
import torchvision.transforms as transforms

from networks.cifarnet import CifarNet
from networks.resnet import resnet32
from networks.vgg import MyVgg

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_normalizer(data_set, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return transforms.Normalize(MEAN, STD)


def get_transformer(data_set, imsize=None, cropsize=None, crop_padding=None, hflip=None):
    transformers = []
    if imsize:
        transformers.append(transforms.Resize(imsize))
    if cropsize:
        transformers.append(transforms.RandomCrop(cropsize, padding=crop_padding))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip(hflip))

    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set))

    return transforms.Compose(transformers)


def get_data_set(args, train_flag=True):
    if args.data_set=="CIFAR10":
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
    else:
        train_transform = get_transformer(args.data_set, args.imsize,
                                  args.cropsize, args.crop_padding, args.hflip)
        test_transform=get_transformer(args.data_set)
    if train_flag:


        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=True,
                                                                transform=train_transform, download=True)
        # data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=True,
                                                                # transform=get_transformer(args.data_set, args.imsize,
                                                                #                           args.cropsize, args.crop_padding, args.hflip), download=True)
    else:
        # data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=False,
        #                                                         transform=get_transformer(args.data_set), download=True)
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=False,
                                                                transform=test_transform, download=True)
    return data_set


def load_network(args):
    network = None
    num_class = 10
    if args.data_set == 'CIFAR100':
        num_class = 100
    if args.network == 'vgg':
        network = MyVgg(num_class=num_class, gated=args.gated, ratio=args.ratio)
    elif args.network == 'resnet':
        network = resnet32(num_class=num_class, gated=args.gated, ratio=args.ratio)
    elif args.network == 'cifarnet':
        network = CifarNet(num_class=num_class, gated=args.gated, ratio=args.ratio)
    if args.load_path:
        check_point = torch.load(args.load_path)
        network.load_state_dict(check_point['state_dict'])
    return network

def band_width_caculation(model_size, rounds, n_clients):


    return model_size*rounds*n_clients

def get_file_size(file_path):
    os.path.getsize(file_path)

    return file_path/1024 #return MB

def save_network(network, args):

    if args.save_path:
        filename = os.path.join(args.save_path, 'ckpt.pth.tar')
        torch.save({'state_dict': network.state_dict()}, filename)
