import torch
import torchvision.transforms as transforms
import torchvision.datasets as dst
import sys
import os
import torchvision.datasets as datasets

sys.path.append("/home/jtcai/SharedSSD/myCode/DML")

from model_cifar.WideResNet import WideResNet_DML, WideResNet
from model_cifar.resnet import ResNet_DML, ResNet
from model_cifar.PyramidNet import PyramidNet_DML
from model_cifar.VGG import VGG_DML
from util import print_network


def define_net_and_opt(args):
    if args.t_name == 'RN':
        tnet = ResNet_DML(num_class=args.num_class, depth=args.t_depth).cuda()
    elif args.t_name == 'WRN':
        tnet = WideResNet_DML(num_class=args.num_class, depth=args.t_depth, widen_factor=args.t_widen_factor).cuda()
    elif args.t_name == 'PYN':
        tnet = PyramidNet_DML(num_class=args.num_class, depth=args.t_depth, alpha=args.t_alpha).cuda()
    elif 'VGG' in args.t_name:
        tnet = VGG_DML(num_class=args.num_class, depth=args.t_depth).cuda()
    else:
        print('Undefined tnet name')

    if args.s_name == 'RN':
        snet = ResNet_DML(num_class=args.num_class, depth=args.s_depth).cuda()
    elif args.s_name == 'WRN':
        snet = WideResNet_DML(num_class=args.num_class, depth=args.s_depth, widen_factor=args.s_widen_factor).cuda()
    elif args.s_name == 'PYN':
        snet = PyramidNet_DML(num_class=args.num_class, depth=args.s_depth, alpha=args.s_alpha).cuda()
    elif 'VGG' in args.s_name:
        snet = VGG_DML(num_class=args.num_class, depth=args.s_depth).cuda()
    else:
        print('Undefined snet name')

    # print_network(net=snet, name='s')
    # print_network(net=tnet, name='t')
    # exit()

    t_fea = tnet.feature_extractor.parameters()
    t_clf = list(tnet.clf1.parameters()) + list(tnet.clf2.parameters())

    s_fea = snet.feature_extractor.parameters()
    s_clf = list(snet.clf1.parameters()) + list(snet.clf2.parameters())

    t_fea_opt = torch.optim.SGD(t_fea, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    t_clf_opt = torch.optim.SGD(t_clf, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    s_fea_opt = torch.optim.SGD(s_fea, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    s_clf_opt = torch.optim.SGD(s_clf, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    nets = {'tnet': tnet, 'snet': snet}
    optimizers = {'t_fea_opt': t_fea_opt, 't_clf_opt': t_clf_opt,
                  's_fea_opt': s_fea_opt, 's_clf_opt': s_clf_opt}

    return nets, optimizers


def define_net_and_opt_1C(args):
    if args.t_name == 'ResNet':
        tnet = ResNet(num_class=args.num_class, depth=args.t_depth).cuda()
    elif args.t_name == 'WideResNet':
        tnet = WideResNet(num_class=args.num_class, depth=args.t_depth, widen_factor=args.t_widen_factor).cuda()
    else:
        print('Undefined tnet name')

    if args.s_name == 'ResNet':
        snet = ResNet(num_class=args.num_class, depth=args.s_depth).cuda()
    elif args.s_name == 'WideResNet':
        snet = WideResNet(num_class=args.num_class, depth=args.s_depth, widen_factor=args.s_widen_factor).cuda()
    else:
        print('Undefined snet name')

    t_opt = torch.optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    s_opt = torch.optim.SGD(snet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    nets = {'tnet': tnet, 'snet': snet}
    optimizers = {'t_opt': t_opt, 's_opt': s_opt}

    return nets, optimizers


def load_dataset(args):
    if args.data_name == 'cifar10':
        print('Using CIFAR10 data!!')
        dataset = dst.CIFAR10
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

    elif args.data_name == 'cifar100':
        print('Using CIFAR100 data!!')
        dataset = dst.CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    else:
        raise Exception('invalid dataset name...')

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    train_loader = torch.utils.data.DataLoader(
        dataset(root=args.dataset_path, transform=train_transform, train=True, download=True),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if args.data_name == 'cifar100':
        test_transform = transforms.Compose([
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        test_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    test_loader = torch.utils.data.DataLoader(
        dataset(root=args.dataset_path, transform=test_transform, train=False, download=True),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def load_python_name(args, python_name):
    name = '{}{}-{}{}-{}'.format(args.t_name, args.t_depth, args.s_name, args.s_depth, args.pseudo_label_type)

    if args.using_intra:
        name = name + '(intra)'

    if '_F_FC_' in python_name:
        python_name = name + '-(F,FC)'
    elif '_F_C_' in python_name:
        python_name = name + '-(F,C)'
    elif '_F_' in python_name:
        python_name = name + '-(F)'
    elif '2C' in python_name:
        python_name = '{}{}-{}{}-2C'.format(args.t_name, args.t_depth, args.s_name, args.s_depth)
    else:
        pass
    return python_name
