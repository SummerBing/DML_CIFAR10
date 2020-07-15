import torch
import torchvision.transforms as transforms
import torchvision.datasets as dst
import sys
import os
from PIL import Image
from model_cifar.WideResNet import WideResNet_DML, WideResNet
from model_cifar.resnet import ResNet_DML, ResNet
from model_cifar.PyramidNet import PyramidNet_DML
from model_cifar.VGG import VGG_DML
from util import print_network


def define_net_and_opt_new(args):
    '''
    Returns:
        models: a list of models
        optimizers: a list of dicts of format {'feature': opt_feature, 'clf': opt_clf}
    '''
    num_model = args.num_model
    
    if args.t_name == 'RN':
        models = [ResNet_DML(num_class=args.num_class, depth=args.t_depth, num_clf=args.num_clf).cuda() for _ in range(num_model)]
    elif args.t_name == 'WRN':
        models = [WideResNet_DML(num_class=args.num_class, depth=args.t_depth, widen_factor=args.t_widen_factor, num_clf=args.num_clf).cuda() for _ in range(num_model)]
    elif args.t_name == 'PYN':
        models = [PyramidNet_DML(num_class=args.num_class, depth=args.t_depth, alpha=args.t_alpha, num_clf=args.num_clf).cuda() for _ in range(num_model)]
    elif 'VGG' in args.t_name:
        models = [VGG_DML(num_class=args.num_class, depth=args.t_depth, num_clf=args.num_clf).cuda() for _ in range(num_model)]
    else:
        raise ValueError('Unknown model type')

    def get_optimizers(model):
        opt_feature = torch.optim.SGD(model.feature_extractor.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        clf_param = []
        for clf in model.clfs:
            clf_param += list(clf.parameters())
        opt_clf = torch.optim.SGD(clf_param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        return {'feature': opt_feature, 'clf': opt_clf}

    optimizers = [get_optimizers(model) for model in models]

    return models, optimizers
        

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
    # print_network(tnet, 'tnet')
    # print_network(snet, 'snet')
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
    optimizers = {'t_fea_opt': t_fea_opt, 't_clf_opt': t_clf_opt, 's_fea_opt': s_fea_opt, 's_clf_opt': s_clf_opt}
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

class MyCIFAR10(dst.CIFAR10):
    '''A dataset that allows multiple augmentated output'''

    def __init__(self, root, train=True, transform=None, download=False, num_augmentation=1):
        super(MyCIFAR10, self).__init__(root=root, train=train, transform=transform, download=download)

        self.num_augmentation = num_augmentation

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            if self.num_augmentation == 1:
                img = self.transform(img)
            else:
                imgs = [self.transform(img) for _ in range(self.num_augmentation)]
                img = torch.stack(imgs, dim=0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MyCIFAR100(dst.CIFAR100):
    '''A dataset that allows multiple augmentated output'''

    def __init__(self, root, train=True, transform=None, download=False, num_augmentation=1):
        super(MyCIFAR100, self).__init__(root=root, train=train, transform=transform, download=download)

        self.num_augmentation = num_augmentation

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            if self.num_augmentation == 1:
                img = self.transform(img)
            else:
                imgs = [self.transform(img) for _ in range(self.num_augmentation)]
                img = torch.stack(imgs, dim=0) # (num_augmentation, img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def load_dataset(args):
    if args.data_name == 'cifar10':
        print('Using CIFAR10 data!!')
        dataset = MyCIFAR10
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    elif args.data_name == 'cifar100':
        print('Using CIFAR100 data!!')
        dataset = MyCIFAR100
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
        dataset(root=args.dataset_path, transform=train_transform, train=True, download=True, num_augmentation=args.num_augmentation),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_transform = transforms.Compose([
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    test_loader = torch.utils.data.DataLoader(
        dataset(root=args.dataset_path, transform=test_transform, train=False, download=True, num_augmentation=1),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def load_python_name(args, python_name):
    name = '{}{}-{}{}-{}'.format(args.t_name, args.t_depth, args.s_name, args.s_depth, args.pseudo_label_type)
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
