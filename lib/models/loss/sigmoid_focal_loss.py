#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Loss function for Image Classification.


import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    def __init__(self, configer):
        super(SigmoidFocalLoss, self).__init__()
        self.params_dict = dict()
        if 'sigmoid_focal_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['sigmoid_focal_loss']

        self.alpha = self.params_dict['alpha']
        self.gamma = self.params_dict['gamma']
        self.norm = self.params_dict['norm']
        self.reduction = self.params_dict['reduction'] if 'reduction' in self.params_dict else 'batchmean'

    def forward(self, inputs, targets):
        onehot_labels = torch.zeros_like(inputs)
        onehot_labels.scatter_(1, torch.unsqueeze(targets, 1), 1.0)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, onehot_labels, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.reduction == 'batchmean':
            return torch.sum(loss) / (inputs.shape[0]*self.norm)
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise Exception('Not implemented!') 
