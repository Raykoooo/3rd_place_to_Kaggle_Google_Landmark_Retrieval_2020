#!/usr/bin/env python
# -*- coding:utf-8 -*-
# origined from https://github.com/KevinMusgrave/pytorch-metric-learning


import torch.nn as nn
from pytorch_metric_learning import losses, miners


class MultiSimilarityLoss(nn.Module):
    def __init__(self, configer):
        super(MultiSimilarityLoss, self).__init__()
        self.params_dict = dict()
        if 'multi_similarity_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['multi_similarity_loss']
        loss_function = losses.MultiSimilarityLoss(alpha=self.params_dict['alpha'], beta=self.params_dict['beta'], base=self.params_dict['base'])
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1) if self.params_dict['miner'] else None
        self.loss_function = losses.CrossBatchMemory(loss_function, self.params_dict['feat_dim'], memory_size=self.params_dict['memory_size'], miner=self.miner) if self.params_dict['xbm'] else loss_function

    def forward(self, inputs, targets):
        if self.params_dict['xbm']:
            loss = self.loss_function(inputs, targets, input_indices_tuple=None)
        elif self.miner is not None:
            indices_tuple = self.miner(inputs, targets)
            loss = self.loss_function(inputs, targets, indices_tuple=indices_tuple)
        else:
            loss = self.loss_function(inputs, targets, indices_tuple=None)
        return loss
