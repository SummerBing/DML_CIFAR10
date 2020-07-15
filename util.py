from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_network(net, name):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number parameters of {} : {}'. format(name, num_params))


def load_pretrained_model(model, pretrained_dict, wfc=True):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if wfc:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and ('fc' not in k))}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def transform_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


def save_checkpoint(state, filename):
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # transposition
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cpu_gpu(use_gpu, data_tensor, volatile=False):
    if use_gpu:
        data = Variable(data_tensor.cuda(), volatile=volatile)
    else:
        data = Variable(data_tensor, volatile=volatile)
    return data


def pkt_cosine_similarity_loss(output_s, output_t, eps=1e-5):
    # out_s: (16, 3, 32, 32)
    # out_t: (16, 3, 32, 32)
    # Normalize each vector by its norm
    # output_s/output_t: (batch_size, num_class)
    output_s = output_s.view(output_s.size(0), -1)  # (16, 3*32*32) [-1, 1]
    output_s_norm = torch.sqrt(torch.sum(output_s ** 2, dim=1, keepdim=True))  # (16, 1)
    output_s = output_s / (output_s_norm + eps)   # Normalization  Add Xi**2 in each row up to 1.
    output_s[output_s != output_s] = 0  #

    output_t = output_t.view(output_t.size(0), -1)
    output_t_norm = torch.sqrt(torch.sum(output_t ** 2, dim=1, keepdim=True))
    output_t = output_t / (output_t_norm + eps)
    output_t[output_t != output_t] = 0  #

    # Calculate the cosine similarity
    output_s_cos_sim = torch.mm(output_s, output_s.transpose(0, 1)) # (16, 16)
    output_t_cos_sim = torch.mm(output_t, output_t.transpose(0, 1))

    # Scale cosine similarity to [0,1]
    output_s_cos_sim = (output_s_cos_sim + 1.0) / 2.0
    output_t_cos_sim = (output_t_cos_sim + 1.0) / 2.0

    # Transform them into probabilities
    output_s_cond_prob = output_s_cos_sim / torch.sum(output_s_cos_sim, dim=1, keepdim=True)
    output_t_cond_prob = output_t_cos_sim / torch.sum(output_t_cos_sim, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(output_t_cond_prob * torch.log((output_t_cond_prob + eps) / (output_s_cond_prob + eps)))

    return loss


def get_pseudo_loss(preds1, preds2, consistency_meters, normalize='reverse', consistency_type='L1', loss_type='KL',
                    weighted_loss=True, adaptive=True, individual_update=True, T=1, threshold=2, softmax_pseudo_label=True):
    # weight calculation
    preds1d = [pred.detach() for pred in preds1]
    preds2d = [pred.detach() for pred in preds2]

    weights = []
    if adaptive:
        consistencys = []  # cal intra-consistency for each network
        for pred1d, pred2d, consistency_meter in zip(preds1d, preds2d, consistency_meters):
            if consistency_type == 'KL':
                consistency = 0.5 * F.kl_div(F.log_softmax(pred1d, dim=1), F.softmax(pred2d, dim=1), reduction='none') +\
                    0.5 * F.kl_div(F.log_softmax(pred2d, dim=1), F.softmax(pred1d, dim=1), reduction='none')
                # if reduction=`mean`, consistency.size() = [batch]
                # if `none`, consistency.size() = [batch, 100]
            elif consistency_type == 'L1':
                consistency = torch.abs(F.softmax(pred1d, dim=1) - F.softmax(pred2d, dim=1))  # [batch, 100]
            elif consistency_type == 'soft_L1':
                consistency = torch.abs(F.softmax(pred1d/3.0, dim=1) - F.softmax(pred2d, dim=1))  # [batch, 100]
            else:
                raise NotImplementedError

            consistencys.append(consistency)  # N个[batch, 100]
            consistency_meter.update(torch.mean(consistency).item(), pred1d.shape[0])
            # consistency_meter: sum, avg, count of current sample's consistency
        min_consistency = min([consistency_meter.avg for consistency_meter in consistency_meters])  # a value
        for consistency, consistency_meter in zip(consistencys, consistency_meters):
            mean_consistency = torch.mean(consistency, dim=1)  # [batch]
            # print(consistency_meter.avg) # a value
            if normalize == 'exp':
                weight = torch.exp((-1 / min_consistency) * mean_consistency)
            elif normalize == 'exp_mean':
                weight = torch.exp((-1 / min_consistency) * (mean_consistency + 0.1 * consistency_meter.avg))
            elif normalize == 'reverse':
                weight = min_consistency / (mean_consistency + 0.1 * consistency_meter.avg)
            elif normalize == 'reverse square':
                weight = (min_consistency ** 2) / (torch.pow(mean_consistency, 2) + 0.1 * (consistency_meter.avg ** 2))
            else:
                raise NotImplementedError
            weights.append(weight)  # weight.size()=[batch], weights.size()=[[batch], [batch]]

    else:
        raise NotImplementedError

    # normalize weights to sum to 1
    weights = torch.stack(weights, dim=0)  # [2, batch], 2 refer to there are 2 group-classifiers
    weights_sum = torch.sum(weights, dim=0)  # [batch]
    weights = weights / weights_sum  # [2, batch], 针对batch里的每一个sample，来自N1、N2的weight和为1
    # compute loss weight
    if weighted_loss:
        weights_sum[weights_sum > threshold] = threshold
        loss_weight = weights_sum / threshold
    # get weighted pseudo label
    if softmax_pseudo_label:
        predsavgd = [torch.softmax((pred1d+pred2d)/2, dim=1) for pred1d, pred2d in zip(preds1d, preds2d)]
    else:
        predsavgd = [(pred1d+pred2d)/2 for pred1d, pred2d in zip(preds1d, preds2d)]  # [batch, 100]
    pseudo = 0
    for predavgd, weight in zip(predsavgd, weights):  # weight.size()=[batch], weight.unsqueeze(1).size()=[batch, 1]
        pseudo += torch.mul(weight.unsqueeze(1), predavgd)
    pseudo.detach_()  # pseudo_label: [batch, 100]

    # update
    loss = 0
    if individual_update:
        for preds in [preds1, preds2]:
            for pred in preds:
                if weighted_loss:
                    if loss_type == 'KL':
                        if softmax_pseudo_label:
                            loss_ = torch.mean(F.kl_div(F.log_softmax(pred, dim=1), pseudo, reduction='none'), dim=1) # [batch]
                            loss += torch.mean(loss_weight * loss_)  # a value
                        else:
                            loss += torch.mean(loss_weight * torch.mean(F.kl_div(F.log_softmax(pred, dim=1), torch.softmax(pseudo, dim=1), reduction='none'), dim=1))
                    elif loss_type == 'KL_soft':  # only softmax_pseudo_label=False
                            loss += torch.mean(loss_weight * torch.mean(
                                F.kl_div(F.log_softmax(pred/3.0, dim=1), torch.softmax(pseudo/3.0, dim=1), reduction='none'), dim=1) * (3.0 * 3.0))
                    else:
                        assert loss_type == 'CE'
                        loss += 0.05 * torch.mean(loss_weight * F.cross_entropy(pred, torch.argmax(pseudo, dim=1)))
                else:
                    loss += F.kl_div(F.log_softmax(pred, dim=1), torch.softmax(pseudo, dim=1))
    else:  # average update
        predsavg = [(pred1+pred2)/2 for pred1, pred2 in zip(preds1, preds2)]
        for predavg in predsavg:
            if weighted_loss:
                if loss_type == 'KL':
                    loss += torch.mean(loss_weight * torch.mean(F.kl_div(F.log_softmax(predavg / T, dim=1), F.softmax(pseudo / T, dim=1), reduction='none'), dim=1)) * (T * T)
                else:
                    assert loss_type == 'CE'
                    loss += 0.05 * torch.mean(loss_weight * F.cross_entropy(predavg, torch.argmax(pseudo, dim=1), reduction='none'))
                loss += F.kl_div(F.log_softmax(predavg / T, dim=1), F.softmax(pseudo / T, dim=1)) * (T * T)
    return pseudo, loss


def get_pseudo_loss_new(preds1, preds2, consistency_meters, smooth=0.1, weight_clip=False, pseudo_soft=True):
    # preds1d = [pred.detach() for pred in preds1]  # [out1_t, out1_s]
    # preds2d = [pred.detach() for pred in preds2]  # [out2_t, out2_s]
    preds1d = preds1
    preds2d = preds2
    consistencys, weights = [], []

    for pred1d, pred2d, consistency_meter in zip(preds1d, preds2d, consistency_meters):
        consistency = torch.abs(F.softmax(pred1d, dim=1) - F.softmax(pred2d, dim=1))
        consistencys.append(consistency)
        consistency_meter.update(torch.mean(consistency).item(), pred1d.shape[0])

    min_consistency = min([consistency_meter.avg for consistency_meter in consistency_meters])
    for consistency, consistency_meter in zip(consistencys, consistency_meters):
        mean_consistency = torch.mean(consistency, dim=1)  # [N]
        weight = min_consistency / (mean_consistency + smooth * consistency_meter.avg)
        weights.append(weight)

    # normalize weights to sum to 1
    weights = torch.stack(weights, dim=0)
    weights_sum = torch.sum(weights, dim=0)
    weights = weights / weights_sum

    # compute loss weight
    loss_weight = weights_sum / len(preds1)
    if weight_clip:  # for numerical stability
        loss_weight[loss_weight > 1] = 1

    # get pseudo label
    predsavgd = [(pred1d + pred2d) / 2 for pred1d, pred2d in zip(preds1d, preds2d)]
    pseudo = 0
    for predavgd, weight in zip(predsavgd, weights):
        pseudo += torch.mul(weight.unsqueeze(1), predavgd)
    pseudo.detach_()

    loss = 0
    for preds in [preds1, preds2]:
        for pred in preds:
            if pseudo_soft:
                loss += torch.mean(loss_weight * torch.mean(F.kl_div(F.log_softmax(pred/3.0, dim=1),
                                                                 torch.softmax(pseudo/3.0, dim=1),
                                                                 reduction='none') * (3.0 * 3.0) / preds1d[0].size(0), dim=1)) * 100
            else:
                loss += torch.mean(loss_weight * torch.mean(F.kl_div(F.log_softmax(pred, dim=1),
                                                                     torch.softmax(pseudo, dim=1),
                                                                     reduction='none'), dim=1))
    return pseudo, loss


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))


def soft_discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1 / 3.0, dim=1) - F.softmax(out2 / 3.0, dim=1)))


def view(input):
    return input.view(input.size(0), -1)


def get_intra_loss(preds, type='l1'):
    assert type in ['l1', 'soft_l1']
    loss = 0
    for pred in preds:
        for i in range(len(pred)):
            for j in range(i+1, len(pred)):
                if type == 'l1':
                    loss += discrepancy(pred[i], pred[j])
                elif type == 'soft_l1':
                    loss += soft_discrepancy(pred[i], pred[j])
                else:
                    raise NotImplementedError
    return loss


def get_mean(pred):
    mean = 0
    for p in pred:
        mean += p
    return mean / len(pred)


def get_inter_loss(preds, type='l1'):
    assert type in ['l1', 'soft_l1']
    loss = 0
    
    means = [get_mean(pred) for pred in preds]
    for i in range(len(means)):
        for j in range(i+1, len(means)):
            if type=='l1':
                loss += discrepancy(means[i], means[j])
            elif type == 'soft_l1':
                loss += soft_discrepancy(means[i], means[j])
            else:
                raise NotImplementedError
    return loss


def get_ensemble_loss(preds, mode='average', type='kl', T=3.0):
    '''
        mode: average or batch or individual. The type of weights for ensemble label calculation.
        type: loss type
    '''
    means = [get_mean(pred) for pred in preds]
    
    if mode == 'average':
        ensemble = sum(means) / len(means)
    else:
        raise NotImplementedError

    loss = 0
    if type == 'kl':
        for pred in preds:
            for p in pred:
                loss += F.kl_div(F.log_softmax(p / T, dim=1), F.softmax(ensemble.detach() / T, dim=1),
                             reduction='mean') * (T * T) 

    return loss, ensemble


def separate_forward(models, images):
    outputs = []
    index = 0
    for model in models:
        model_outputs = []
        for clf in model.clfs:
            image = images[:,index,...]
            output = clf(model.feature_extractor(image))
            model_outputs.append(output)
            index += 1
        outputs.append(model_outputs)

    return outputs
