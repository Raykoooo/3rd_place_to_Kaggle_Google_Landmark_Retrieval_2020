#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Loss function for Image Classification.

import torch
import torch.nn as nn
from lib.models.loss import (CELoss, HardTripletLoss, KLLoss,
                             LiftedStructureLoss, PrecisionAtTopkLoss,
                             ProxyNCALoss, SigmoidFocalLoss, SoftmaxFocalLoss,
                             TripletMarginLoss, MultiSimilarityLoss, CircleLoss,
                             OnlineTripletMarginLoss, SmoothAPLoss)

BASE_LOSS_DICT = dict(
    ce_loss=0,
    kl_loss=1,
    hard_triplet_loss=2,
    lifted_structure_loss=3,
    precision_at_topk_loss=4,
    proxy_nca_loss=5,
    sigmoid_focal_loss=6,
    softmax_focal_loss=7,
    triplet_margin_loss=8,
    multi_similarity_loss=9,
    circle_loss=10,
    online_triplet_margin_loss=11,
    smooth_ap_loss=12,
)


class Loss(nn.Module):
    def __init__(self, configer):
        super(Loss, self).__init__()
        self.configer = configer
        self.func_list = [CELoss(self.configer), KLLoss(self.configer),
                          HardTripletLoss(self.configer), LiftedStructureLoss(self.configer),
                          PrecisionAtTopkLoss(self.configer), ProxyNCALoss(self.configer),
                          SigmoidFocalLoss(self.configer), SoftmaxFocalLoss(self.configer),
                          TripletMarginLoss(self.configer), MultiSimilarityLoss(self.configer),
                          CircleLoss(self.configer), OnlineTripletMarginLoss(self.configer),
                          SmoothAPLoss(self.configer)]

    def forward(self, out_list):
        loss_dict = out_list[-1]
        out_dict = dict()
        weight_dict = dict()
        for key, item in loss_dict.items():
            out_dict[key] = self.func_list[int(item['type'].float().mean().item())](*item['params'])
            weight_dict[key] = item['weight'].mean().item()

        loss = 0.0
        for key in out_dict:
            loss += out_dict[key] * weight_dict[key]

        out_dict['loss'] = loss
        return out_dict, weight_dict
