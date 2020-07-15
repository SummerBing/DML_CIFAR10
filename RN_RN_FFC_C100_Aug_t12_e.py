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

sys.path.append("/home/jtcai/SharedSSD/myCode/DML")
from util import AverageMeter, accuracy, transform_time, save_checkpoint, \
    cpu_gpu, discrepancy, soft_discrepancy, get_pseudo_loss_new, get_pseudo_loss
from C100_12 import load_setting_aug_12

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
parser = argparse.ArgumentParser(description='DML')
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--epoch_list', type=list, default=[150, 75, 75])
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')

parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--data_name', type=str, default='cifar10')
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--dataset_path', type=str, default='/home/jtcai/SharedSSD/myCode/DML/dataset/')

parser.add_argument('--t_name', default='RN-12')
parser.add_argument('--t_depth', type=int, default=32)
# parser.add_argument('--t_name', default='WRN-12')
# parser.add_argument('--t_depth', type=int, default=16)
parser.add_argument('--t_widen_factor', type=int, default=2)  # [WRN]
parser.add_argument('--t_alpha', type=int, default=240)  # for pyramidnet

parser.add_argument('--s_name', default='RN-12')
parser.add_argument('--s_depth', type=int, default=32)
# parser.add_argument('--s_name', default='WRN-12')
# parser.add_argument('--s_depth', type=int, default=16)
parser.add_argument('--s_widen_factor', type=int, default=2)
parser.add_argument('--s_alpha', type=int, default=240)

parser.add_argument('--TIMES', type=int, default=6)
parser.add_argument('--Begin_TIMES', type=int, default=1)

parser.add_argument('--pseudo_label_type', default='[t12-e]')  # [`Ensemble`, `New`, `Old`]
pseudo_burnin = 'linear'  # exp, linear, none


intra_t_ratio = intra_s_ratio = 100
kl_loss_t = kl_loss_s = 100
pseudo_loss_ratio = 100
inter_ratio = 100
inter_intra_step = pseudo_step = 1

using_soft_discrepancy = True  # [L1, soft_L1]
inter_KL = False
inter_burnin = 'none'  # exp, linear, none


def main():
    global args
    args = parser.parse_args()
    if args.cuda:
        cudnn.benchmark = True

    python_name = load_setting_aug_12.load_python_name(args, Path(__file__).name)
    print(python_name)
    global save_max_accu
    save_max_accu = './{}.txt'.format(python_name)
    with open(save_max_accu, 'a') as f:
        f.write(python_name)
        f.write('\n')
        f.close()

    for TIMES in range(1, args.TIMES):
        train_loader, test_loader = load_setting_aug_12.load_dataset_Aug_t12(args)
        nets, optimizers = load_setting_aug_12.define_net_and_opt(args)
        max_prec = [0.0, 0.0, 0.0, 0.0, 0.0]

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            adjust_lr(optimizers, epoch, times=TIMES)
            train(train_loader, nets, optimizers, epoch)
            epoch_time = time.time() - epoch_start_time
            print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

            print('testing the models......')
            pred_list = test12(test_loader, nets, epoch)
            for id in range(len(pred_list)):
                if max_prec[id] < pred_list[id]:
                    max_prec[id] = pred_list[id]
            current_max = 'Net:{} || Current-Max:[{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
                python_name, max_prec[0], max_prec[1], max_prec[2], max_prec[3], max_prec[4])
            print(current_max)
            if epoch == args.epochs:
                with open(save_max_accu, 'a') as f:
                    f.write(str(current_max))
                    f.write('\n')
                    f.close()


def train(train_loader, nets, optimizers, epoch, mode=0):
    clf_losses = [AverageMeter() for _ in range(5)]
    intra_losses = [AverageMeter() for _ in range(2)]
    inter_losses = [AverageMeter() for _ in range(4)]
    top1 = [AverageMeter() for _ in range(5)]
    consistency_meters = [AverageMeter() for _ in range(2)]
    pseudo_losses = [AverageMeter() for _ in range(2)]
    weight = [AverageMeter() for _ in range(2)]

    tnet = nets['tnet']
    snet = nets['snet']
    tnet.train()
    snet.train()

    t_fea_opt = optimizers['t_fea_opt']
    t_clf_opt = optimizers['t_clf_opt']
    s_fea_opt = optimizers['s_fea_opt']
    s_clf_opt = optimizers['s_clf_opt']

    for idx, (img1, img2, target) in enumerate(train_loader, start=1):
        img1 = cpu_gpu(args.cuda, img1, volatile=False)
        img2 = cpu_gpu(args.cuda, img2, volatile=False)
        target = cpu_gpu(args.cuda, target, volatile=False)

        out1_t, out2_t = tnet(img1, img2)
        out1_s, out2_s = snet(img1, img2)
        cls_t1 = F.cross_entropy(out1_t, target)
        cls_t2 = F.cross_entropy(out2_t, target)
        cls_s1 = F.cross_entropy(out1_s, target)
        cls_s2 = F.cross_entropy(out2_s, target)

        loss1 = cls_t1 + cls_t2
        reset_grad([t_fea_opt, t_clf_opt])
        loss1.backward()
        set_step([t_fea_opt, t_clf_opt])
        loss2 = cls_s1 + cls_s2
        reset_grad([s_fea_opt, s_clf_opt])
        loss2.backward()
        set_step([s_fea_opt, s_clf_opt])

        for kk in range(inter_intra_step):
            out1_t, out2_t = tnet(img1, img2)
            out1_s, out2_s = snet(img1, img2)
            cls_t1 = F.cross_entropy(out1_t, target)
            cls_t2 = F.cross_entropy(out2_t, target)
            cls_s1 = F.cross_entropy(out1_s, target)
            cls_s2 = F.cross_entropy(out2_s, target)

            if using_soft_discrepancy:
                intra_t = soft_discrepancy(out1_t, out2_t) * intra_t_ratio  # a value
                intra_s = soft_discrepancy(out1_s, out2_s) * intra_s_ratio
            else:
                intra_t = discrepancy(out1_t, out2_t) * intra_t_ratio
                intra_s = discrepancy(out1_s, out2_s) * intra_s_ratio

            loss_1 = intra_t
            reset_grad([t_fea_opt])
            loss_1.backward()
            set_step([t_fea_opt])
            loss_2 = intra_s
            reset_grad([s_fea_opt])
            loss_2.backward()
            set_step([s_fea_opt])

        for jj in range(pseudo_step):
            out1_t, out2_t = tnet(img1, img2)
            out1_s, out2_s = snet(img1, img2)
            if inter_KL:
                inter_t1 = F.kl_div(F.log_softmax(out1_t / 3.0, dim=1), F.softmax(out1_s.detach() / 3.0, dim=1),
                                    reduction='mean') * (3.0 * 3.0) / img1.size(0) * inter_ratio
                inter_t2 = F.kl_div(F.log_softmax(out2_t / 3.0, dim=1), F.softmax(out2_s.detach() / 3.0, dim=1),
                                    reduction='mean') * (3.0 * 3.0) / img1.size(0) * inter_ratio

                inter_s1 = F.kl_div(F.log_softmax(out1_s / 3.0, dim=1), F.softmax(out1_t.detach() / 3.0, dim=1),
                                    reduction='mean') * (3.0 * 3.0) / img1.size(0) * inter_ratio
                inter_s2 = F.kl_div(F.log_softmax(out2_s / 3.0, dim=1), F.softmax(out2_t.detach() / 3.0, dim=1),
                                    reduction='mean') * (3.0 * 3.0) / img1.size(0) * inter_ratio
            elif using_soft_discrepancy:
                inter_t1 = soft_discrepancy(out1_t, out1_s.detach()) * inter_ratio
                inter_t2 = soft_discrepancy(out2_t, out2_s.detach()) * inter_ratio
                inter_s1 = soft_discrepancy(out1_s, out1_t.detach()) * inter_ratio
                inter_s2 = soft_discrepancy(out2_s, out2_t.detach()) * inter_ratio
            else:
                inter_t1 = discrepancy(out1_t, out1_s.detach()) * inter_ratio
                inter_t2 = discrepancy(out2_t, out2_s.detach()) * inter_ratio
                inter_s1 = discrepancy(out1_s, out1_t.detach()) * inter_ratio
                inter_s2 = discrepancy(out2_s, out2_t.detach()) * inter_ratio

            if using_soft_discrepancy:
                intra_t = soft_discrepancy(out1_t, out2_t) * intra_t_ratio  # a value
                intra_s = soft_discrepancy(out1_s, out2_s) * intra_s_ratio
            else:
                intra_t = discrepancy(out1_t, out2_t) * intra_t_ratio
                intra_s = discrepancy(out1_s, out2_s) * intra_s_ratio
            intra_t_e = torch.exp(-intra_t)
            intra_s_e = torch.exp(-intra_s)
            length = intra_t_e + intra_s_e
            wt = intra_t_e / length
            ws = intra_s_e / length

            mean_t = (out1_t + out2_t) * 0.5
            mean_s = (out1_s + out2_s) * 0.5
            pseudo = wt * mean_t.detach() + ws * mean_s.detach()
            kl_t1 = F.kl_div(F.log_softmax(out1_t / 3.0, dim=1), F.softmax(pseudo.detach() / 3.0, dim=1),
                             reduction='mean') * (3.0 * 3.0) / img1.size(0) * kl_loss_t
            kl_t2 = F.kl_div(F.log_softmax(out2_t / 3.0, dim=1), F.softmax(pseudo.detach() / 3.0, dim=1),
                             reduction='mean') * (3.0 * 3.0) / img1.size(0) * kl_loss_t
            kl_s1 = F.kl_div(F.log_softmax(out1_s / 3.0, dim=1), F.softmax(pseudo.detach() / 3.0, dim=1),
                             reduction='mean') * (3.0 * 3.0) / img1.size(0) * kl_loss_s
            kl_s2 = F.kl_div(F.log_softmax(out2_s / 3.0, dim=1), F.softmax(pseudo.detach() / 3.0, dim=1),
                             reduction='mean') * (3.0 * 3.0) / img1.size(0) * kl_loss_s
            pseudo_loss_t = kl_t1 + kl_t2
            pseudo_loss_s = kl_s1 + kl_s2

            if pseudo_burnin == 'exp':
                beta = 2 / (1 + math.exp(-10 * epoch / args.epochs)) - 1
            elif pseudo_burnin == 'linear':
                beta = min(1, 1.25 * (epoch / args.epochs))
            else:
                assert pseudo_burnin == 'none'
                beta = 1
            pseudo_cls = F.cross_entropy(pseudo, target)
            pseudo_loss_t = pseudo_loss_t * beta * pseudo_loss_ratio
            pseudo_loss_s = pseudo_loss_s * beta * pseudo_loss_ratio

            loss_2t = inter_t1 + inter_t2 + pseudo_loss_t
            reset_grad([t_clf_opt, t_fea_opt])
            loss_2t.backward()
            set_step([t_clf_opt, t_fea_opt])

            loss_2s = inter_s1 + inter_s2 + pseudo_loss_s
            reset_grad([s_clf_opt, s_fea_opt])
            loss_2s.backward()
            set_step([s_clf_opt, s_fea_opt])

        out = [out1_t, out2_t, out1_s, out2_s, pseudo]
        accu = [accuracy(pred, target, topk=(1,))[0] for pred in out]
        for acc, top in zip(accu, top1):
            top.update(acc, img1.size(0))

        cls_loss = [cls_t1, cls_t2, cls_s1, cls_s2, pseudo_cls]
        for loss, losses in zip(cls_loss, clf_losses):
            losses.update(loss.item(), img1.size(0))

        intra_list = [intra_t, intra_s]
        for intra, intra_meter in zip(intra_list, intra_losses):
            intra_meter.update(intra.item(), img1.size(0))
        inter_list = [inter_t1, inter_t2, inter_s1, inter_s2]
        for inter, inter_meter in zip(inter_list, inter_losses):
            inter_meter.update(inter.item(), img1.size(0))
        pseudo_losses[0].update(pseudo_loss_t, img1.size(0))
        pseudo_losses[1].update(pseudo_loss_s, img1.size(0))
        weight[0].update(wt.item())
        weight[1].update(ws.item())

        if idx % args.print_freq == 0:
            result = 'Epoch:{}, cls-loss:({:.3f},{:.3f},{:.3f},{:.3f},{:.3f}), ' \
                     'intra-loss:({:.3f},{:.3f}), inter-loss:({:.3f},{:.3f},{:.3f},{:.3f}), ' \
                     'pseudo-loss:({:.3f},{:.3f}), weight:({:.4f},{:.4f})'.format(
                epoch, clf_losses[0].avg, clf_losses[1].avg, clf_losses[2].avg, clf_losses[3].avg, clf_losses[4].avg,
                intra_losses[0].avg, intra_losses[1].avg,
                inter_losses[0].avg, inter_losses[1].avg, inter_losses[2].avg, inter_losses[3].avg,
                pseudo_losses[0].avg, pseudo_losses[1].avg, weight[0].avg, weight[1].avg)
            print(result)
            result1 = 'Epoch:{}, top1:({:.4f},{:.4f},{:.4f},{:.4f},{:.4f})'.format(
                epoch, top1[0].avg, top1[1].avg, top1[2].avg, top1[3].avg, top1[4].avg)
            print(result1)


def test12(test_loader, nets, epoch):
    clf_losses = [AverageMeter() for _ in range(5)]
    top1 = [AverageMeter() for _ in range(5)]
    consistency_meters = [AverageMeter() for _ in range(2)]
    weight = [AverageMeter() for _ in range(2)]

    snet = nets['snet']
    tnet = nets['tnet']
    snet.eval()
    tnet.eval()

    for idx, (img1, img2, target) in enumerate(test_loader, start=1):
        img1 = cpu_gpu(args.cuda, img1, volatile=True)
        img2 = cpu_gpu(args.cuda, img2, volatile=True)
        target = cpu_gpu(args.cuda, target, volatile=True)
        out1_t, out2_t = tnet(img1, img2)
        out1_s, out2_s = snet(img1, img2)
        cls_t1 = F.cross_entropy(out1_t, target)
        cls_t2 = F.cross_entropy(out2_t, target)
        cls_s1 = F.cross_entropy(out1_s, target)
        cls_s2 = F.cross_entropy(out2_s, target)

        preds1_t = [out1_t, out1_s]
        preds2_t = [out2_t, out2_s]

        if 'New' in args.pseudo_label_type:
            pseudo, pseudo_loss = get_pseudo_loss_new(preds1_t, preds2_t, consistency_meters, weight_clip=True)
        elif 'Old' in args.pseudo_label_type:
            pseudo, pseudo_loss = get_pseudo_loss(preds1_t, preds2_t, consistency_meters, softmax_pseudo_label=True)
        else:
            mean_t = (out1_t + out2_t) * 0.5
            mean_s = (out1_s + out2_s) * 0.5
            if using_soft_discrepancy:
                intra_t = soft_discrepancy(out1_t, out2_t) * intra_t_ratio
                intra_s = soft_discrepancy(out1_s, out2_s) * intra_s_ratio
            else:
                intra_t = discrepancy(out1_t, out2_t) * intra_t_ratio
                intra_s = discrepancy(out1_s, out2_s) * intra_s_ratio
            intra_t_e = torch.exp(-intra_t)
            intra_s_e = torch.exp(-intra_s)
            length = intra_t_e + intra_s_e
            wt = intra_t_e / length
            ws = intra_s_e / length
            # wt = intra_t
            # ws = intra_s
            pseudo = wt * mean_t.detach() + ws * mean_s.detach()

        pseudo_cls = F.cross_entropy(pseudo, target)

        out = [out1_t, out2_t, out1_s, out2_s, pseudo]
        accu = [accuracy(pred, target, topk=(1,))[0] for pred in out]
        for acc, top in zip(accu, top1):
            top.update(acc, img1.size(0))
        cls_loss = [cls_t1, cls_t2, cls_s1, cls_s2, pseudo_cls]
        for loss, losses in zip(cls_loss, clf_losses):
            losses.update(loss.item(), img1.size(0))
        weight[0].update(wt.item())
        weight[1].update(ws.item())

    result = 'Epoch:{}, cls-loss:({:.3f},{:.3f},{:.3f},{:.3f},{:.3f}), weight:({:.4f},{:.4f}), ' \
             'top1:({:.4f},{:.4f},{:.4f},{:.4f},{:.4f})'. \
        format(epoch, clf_losses[0].avg, clf_losses[1].avg, clf_losses[2].avg, clf_losses[3].avg, clf_losses[4].avg,
               weight[0].avg, weight[1].avg,
               top1[0].avg, top1[1].avg, top1[2].avg, top1[3].avg, top1[4].avg)
    print(result)
    return [top1[0].avg, top1[1].avg, top1[2].avg, top1[3].avg, top1[4].avg]


def adjust_lr(optimizers, epoch, times):
    scale = 0.1
    lr_list = []
    for i in range(len(args.epoch_list)):
        lr_list += [args.lr * math.pow(scale, i)] * args.epoch_list[i]
    lr = lr_list[epoch - 1]
    for key in optimizers:
        for param_group in optimizers[key].param_groups:
            param_group['lr'] = lr
    print('Times:[{}/{}] || Epoch: [{}], lr: {}'.format(times, args.TIMES, epoch, lr))


def reset_grad(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()


def set_step(listt):
    for ll in listt:
        ll.step()


def normalize(factor, eps=1e-5):
    norm = torch.norm(factor.view(factor.size(0), -1), dim=1)
    norm = norm.view(norm.size(0), 1, 1, 1)
    norm_factor = torch.div(factor, norm + eps)
    return norm_factor


if __name__ == '__main__':
    main()
