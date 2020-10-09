#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Image classification running score.


from lib.utils.tools.average_meter import DictAverageMeter


class RunningScore(object):
    def __init__(self, configer):
        self.configer = configer
        self.top1_acc = DictAverageMeter()
        self.top3_acc = DictAverageMeter()
        self.top5_acc = DictAverageMeter()
        self.ignore_index = configer.get('loss.params.ce_loss.ignore_index', default=None)

    def get_top1_acc(self):
        return self.top1_acc.info()

    def get_top3_acc(self):
        return self.top3_acc.info()

    def get_top5_acc(self):
        return self.top5_acc.info()

    def update(self, out_dict, label_dict):
        """Computes the precision@k for the specified values of k"""
        top1_acc_dict = dict()
        top3_acc_dict = dict()
        top5_acc_dict = dict()
        batch_size_dict = dict()
        for key in label_dict.keys():
            output, target = out_dict[key], label_dict[key]
            topk = (1, 3, 5)
            maxk = min(max(topk), output.size(1))
            batch_size_dict[key] = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            target = target.view(1, -1)
            if self.ignore_index:
                keep_idx = target != self.ignore_index
                target = target[keep_idx].reshape(1, -1)
                pred = pred[keep_idx.expand_as(pred)].reshape(maxk, -1)
                batch_size_dict[key] = target.size(1)
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=False)
                res.append(correct_k / batch_size_dict[key])

            top1_acc_dict[key] = res[0].item()
            top3_acc_dict[key] = res[1].item()
            top5_acc_dict[key] = res[2].item()

        self.top1_acc.update(top1_acc_dict, batch_size_dict)
        self.top3_acc.update(top3_acc_dict, batch_size_dict)
        self.top5_acc.update(top5_acc_dict, batch_size_dict)

    def reset(self):
        self.top1_acc.reset()
        self.top3_acc.reset()
        self.top5_acc.reset()
