#!/usr/bin/env python
# -*- coding:utf-8 -*-
# origined from https://github.com/KevinMusgrave/pytorch-metric-learning


import torch.nn as nn
from pytorch_metric_learning import losses
from lib.models.miners import online_triplet_miner as miners 


class OnlineTripletMarginLoss(nn.Module):
    def __init__(self, configer):
        super(OnlineTripletMarginLoss, self).__init__()
        self.params_dict = dict()
        if 'online_triplet_margin_loss' in configer.get('loss', 'params'):
            self.params_dict = configer.get('loss', 'params')['online_triplet_margin_loss']
        self.loss_function = losses.TripletMarginLoss(margin=self.params_dict['margin'], distance_norm=self.params_dict['distance_norm'], power=self.params_dict['power'], swap=self.params_dict['swap'], smooth_loss=self.params_dict['smooth_loss'], avg_non_zero_only=self.params_dict['avg_non_zero_only'], triplets_per_anchor=self.params_dict['triplets_per_anchor'])
        self.miner = miners.OnlineTripletMiner(margin=self.params_dict['margin'])

    def forward(self, inputs, targets):
        indices_tuple = self.miner(inputs, targets)
        loss = self.loss_function(inputs, targets, indices_tuple=indices_tuple)
        return loss
