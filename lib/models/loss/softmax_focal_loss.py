#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Loss function for Image Classification.


import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, configer):
        super(SoftmaxFocalLoss, self).__init__()
        self.params_dict = dict()
        if 'softmax_focal_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['softmax_focal_loss']

        self.alpha = self.params_dict['alpha']
        self.gamma = self.params_dict['gamma']
        self.reduction = self.params_dict['reduction'] if 'reduction' in self.params_dict else 'mean'

    def forward(self, inputs, targets):
        onehot_labels = torch.zeros_like(inputs)
        onehot_labels.scatter_(1, torch.unsqueeze(targets, 1), 1.0)
        pred = F.softmax(inputs, dim=1)
        pt =  (pred*onehot_labels).sum(1).view(-1,1)
        log_pt = pt.log()
        loss = -self.alpha*(torch.pow((1.-pt), self.gamma))*log_pt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise Exception('Not implemented!')
