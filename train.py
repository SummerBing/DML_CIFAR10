from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn

import sys
import argparse
import os
import time
import torch.nn.functional as F
import math
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from util import AverageMeter, accuracy, transform_time, save_checkpoint, cpu_gpu
from util import get_intra_loss, get_inter_loss, get_ensemble_loss, separate_forward
from C10_Aug import load_setting_aug

parser = argparse.ArgumentParser(description='DML')
# parser.add_argument('--print_freq', type=int, default=100)

# training setup
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
# parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        # help='Decrease learning rate at these epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--cuda', type=int, default=1)

# model
parser.add_argument('--t_name', default='RN')  # [RN, WRN, PYN, VGG]
parser.add_argument('--t_depth', type=int, default=32)  # [RN, WRN, VGG]
parser.add_argument('--t_widen_factor', type=int, default=2)  # [WRN]
parser.add_argument('--t_alpha', type=int, default=240)  # for pyramidnet
parser.add_argument('--num_model', type=int, default=2, help='number of models')
parser.add_argument('--num_clf', type=int, default=2, help='number of classifiers in a model')

# data
parser.add_argument('--data_name', type=str, default='cifar100')
parser.add_argument('--num_class', type=int, default=100)
parser.add_argument('--dataset_path', type=str, default='./data/')
parser.add_argument('--diffaug', action='store_true', 
                    help='if ture, use differernt augmentation for different models and classifiers')

# inter intra ensemble
parser.add_argument('--use_intra', dest='use_intra', action='store_true')
parser.add_argument('--no_intra', dest='use_intra', action='store_false')
parser.set_defaults(use_intra=True)
parser.add_argument('--intra_step', type=int, default=1)
parser.add_argument('--intra_ratio', type=float, default=10)
parser.add_argument('--intra_loss_type', type=str, default='soft_l1', choices=['l1', 'soft_l1'])

parser.add_argument('--use_inter', dest='use_inter', action='store_true')
parser.add_argument('--no_inter', dest='use_inter', action='store_false')
parser.set_defaults(use_inter=True)
parser.add_argument('--inter_step', type=int, default=1)
parser.add_argument('--inter_ratio', type=float, default=10)
parser.add_argument('--inter_loss_type', type=str, default='soft_l1', choices=['l1', 'soft_l1'])

parser.add_argument('--use_ensemble', dest='use_ensemble', action='store_true')
parser.add_argument('--no_ensemble', dest='use_ensemble', action='store_false')
parser.set_defaults(use_ensemble=True)
parser.add_argument('--ensemble_step', type=int, default=1)
parser.add_argument('--ensemble_type', type=str, default='kl')
parser.add_argument('--ensemble_ratio', type=float, default=10)
parser.add_argument('--ensemble_mode', type=str, default='average',\
         choices=['average', 'batch_weighted', 'sample_weighted'])
parser.add_argument('--ensemble_temp', type=float, default=3.0)


def main():
    global args
    args = parser.parse_args()
    print('args', args)
    if not args.diffaug:
        args.num_augmentation = 1
    else:
        args.num_augmentation = args.num_model * args.num_clf
    if args.cuda:
        cudnn.benchmark = True

    # python_name = load_setting_aug.load_python_name(args, Path(__file__).name)
    # print(python_name)
    # global save_max_accu
    # save_max_accu = './{}.txt'.format(python_name)

    models, optimizers = load_setting_aug.define_net_and_opt_new(args)
    opt_list = [optimizer['feature'] for optimizer in optimizers] + [optimizer['clf'] for optimizer in optimizers]
    schedulers = [MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1) for optimizer in opt_list]

    train_loader, test_loader = load_setting_aug.load_dataset(args)
    max_prec = [0.0 for _ in range(args.num_model*args.num_clf + 1)]

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        set_step(schedulers)
        train(train_loader, models, optimizers, epoch, args)

        epoch_time = time.time() - epoch_start_time
        # print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))
        # print('testing the models......')
        pred_list = test(test_loader, models, epoch)
        for id in range(len(pred_list)):
            if max_prec[id] < pred_list[id]:
                max_prec[id] = pred_list[id]
        max_str = 'Current-Max:['+ '{:.2f},'*args.num_model*args.num_clf +'{:.2f}]'
        current_max = max_str.format(*max_prec)
        print(current_max)
        # if epoch == args.epochs:
        #     with open(save_max_accu, 'a') as f:
        #         f.write(str(current_max))
        #         f.write('\n')
        #         f.close()


def train(train_loader, models, optimizers, epoch, args):
    # meters for record keeping
    num_clf = args.num_clf
    num_model = args.num_model
    accuracy_meters = [AverageMeter() for _ in range(num_clf*num_model+1)]
    cls_loss_meter, intra_loss_meter, inter_loss_meter, ensemble_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    set_mode(models, mode='train')

    with tqdm(total=len(train_loader)) as tbar:
        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            # basic update with labels
            if not args.diffaug:
                preds = [model(img) for model in models] # list of list
            else:
                preds = separate_forward(models, img, args)

            cls_loss = 0
            for pred in preds:
                for p in pred:
                    cls_loss += F.cross_entropy(p, target)

            opt_list = [optimizer['feature'] for optimizer in optimizers] + [optimizer['clf'] for optimizer in optimizers]
            reset_grad(opt_list)
            cls_loss.backward()
            set_step(opt_list)
            
            # intra inter
            if args.use_intra or args.use_inter:
                for _ in range(args.intra_step):
                    if not args.diffaug:
                        preds = [model(img) for model in models] # list of list
                    else:
                        preds = separate_forward(models, img, args)

                    intra_inter_loss = 0
                    if args.use_intra:
                        intra_loss = get_intra_loss(preds, type=args.intra_loss_type)
                        intra_loss *= args.intra_ratio
                        intra_inter_loss += intra_loss
                    if args.use_inter:
                        inter_loss = get_inter_loss(preds, type=args.inter_loss_type)
                        inter_loss *= args.inter_ratio
                        intra_inter_loss += inter_loss
                    opt_list = [optimizer['feature'] for optimizer in optimizers]
                    reset_grad(opt_list)
                    intra_inter_loss.backward()
                    set_step(opt_list)

            # ensemble
            if args.use_ensemble:
                for _ in range(args.ensemble_step):
                    if not args.diffaug:
                        preds = [model(img) for model in models] # list of list
                    else:
                        preds = separate_forward(models, img, args)

                    ensemble_loss, ensemble_pred = get_ensemble_loss(preds, mode=args.ensemble_mode, type=args.ensemble_type, T=args.ensemble_temp)
                    ensemble_loss *= args.ensemble_ratio

                    opt_list = [optimizer['feature'] for optimizer in optimizers] + [optimizer['clf'] for optimizer in optimizers]
                    reset_grad(opt_list)
                    ensemble_loss.backward()
                    set_step(opt_list)

            out = [p for pred in preds for p in pred]
            if args.use_ensemble:
                out.append(ensemble_pred)
            accu = [accuracy(pred, target, topk=(1,))[0] for pred in out]
            for acc, accuracy_meter in zip(accu, accuracy_meters):
                accuracy_meter.update(acc, preds[0][0].size(0))

            cls_loss_meter.update(cls_loss.item(), preds[0][0].size(0))
            if args.use_intra:
                intra_loss_meter.update(intra_loss.item(), preds[0][0].size(0))
            if args.use_inter:
                inter_loss_meter.update(inter_loss.item(), preds[0][0].size(0))
            if args.use_ensemble:
                ensemble_loss_meter.update(ensemble_loss.item(), preds[0][0].size(0))

            tbar.update()
    result_str = '\nTraining: Epoch:{},cls-loss:({:.3f}),intra-loss:({:.3f}),inter-loss:({:.3f}),pseudo-loss:({:.3f}),'\
                +'accuracy:('+'{:.4f} '*(num_clf*num_model)+ '{:.4f})'
    result = result_str.format(
            epoch, cls_loss_meter.avg, intra_loss_meter.avg, inter_loss_meter.avg, ensemble_loss_meter.avg,
             *[meter.avg for meter in accuracy_meters])
    print(result)

def test(test_loader, models, epoch):
    num_clf = args.num_clf
    num_model = args.num_model
    accuracy_meters = [AverageMeter() for _ in range(num_clf*num_model+1)]

    set_mode(models, mode='eval')

    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            preds = [model(img) for model in models]
            _, ensemble_pred = get_ensemble_loss(preds, mode=args.ensemble_mode, type=args.ensemble_type, T=args.ensemble_temp)
            out = [p for pred in preds for p in pred]
            out.append(ensemble_pred)
            accu = [accuracy(pred, target, topk=(1,))[0] for pred in out]
            for acc, accuracy_meter in zip(accu, accuracy_meters):
                accuracy_meter.update(acc, preds[0][0].size(0))

    result_str = 'Testing: Epoch:{},' + 'accuracy:('+'{:.4f} '*(num_clf*num_model)+ '{:.4f})'
    result = result_str.format(epoch, *[meter.avg for meter in accuracy_meters])
    print(result)
    return [meter.avg for meter in accuracy_meters]


def reset_grad(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()


def set_step(listt):
    for ll in listt:
        ll.step()

def set_mode(models, mode):
    assert mode in ['train', 'eval']
    for model in models:
        if mode=='train':
            model.train()
        else:
            model.eval()

def normalize(factor, eps=1e-5):
    norm = torch.norm(factor.view(factor.size(0), -1), dim=1)
    norm = norm.view(norm.size(0), 1, 1, 1)
    norm_factor = torch.div(factor, norm + eps)
    return norm_factor


if __name__ == '__main__':
    main()
