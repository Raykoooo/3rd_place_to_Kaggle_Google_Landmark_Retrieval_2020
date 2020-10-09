#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ResNet in PyTorch.


import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
# from PIL import Image
# import numpy as np

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class DeployClsModel(nn.Module):
    def __init__(self, configer, flag="", target_class=1):
        super(DeployClsModel, self).__init__()
        self.configer = configer
        self.flag = flag if len(flag) == 0 else "{}_".format(flag)
        self.backbone = BackboneSelector(configer).get_backbone(
            backbone_type=configer.get('network.{}backbone'.format(self.flag)),
            rm_last_stride=configer.get('network', '{}rm_last_stride'.format(self.flag), default=False)
        )
        self.reduction = None
        fc_dim_out = configer.get('network.{}fc_dim'.format(self.flag), default=None)
        fc_dim = self.backbone.num_features
        if fc_dim_out is not None:
            self.reduction = nn.Conv2d(self.backbone.num_features, fc_dim_out, 1)
            fc_dim = fc_dim_out
        self.bn = None
        if configer.get('network.{}fc_bn'.format(self.flag), default=None):
            self.bn = nn.BatchNorm2d(fc_dim)
            
        if self.configer.get('deploy.extract_score', default=False) or self.configer.get('deploy.extract_cam', default=False):
            self.linear_lists = nn.ModuleList()
            for source in range(self.configer.get('data', 'num_data_sources')):
                linear_list = nn.ModuleList()
                linear_type = self.configer.get('network', '{}src{}_linear_type'.format(self.flag, source))
                for num_classes in self.configer.get('data.src{}_num_classes'.format(source)):
                    linear_list.append(ModuleHelper.Linear(linear_type)(fc_dim, num_classes))
                self.linear_lists.append(linear_list)        

    def forward(self, x, extract_score=False, extract_cam=False):
        x = ModuleHelper.preprocess(x, self.configer.get('data.normalize'))
        x = self.backbone(x)
        conv_out = x[:]
        x = ModuleHelper.postprocess(x, method=self.configer.get('deploy', 'pool_type'))
        x = self.reduction(x) if self.reduction else x
        x = self.bn(x) if self.bn else x
        x = x.flatten(1)
        out = ModuleHelper.normalize(x, method=self.configer.get('deploy', 'norm_type'))

        if extract_cam:
            logits = self.linear_lists[0][0](x, None)
            score = ModuleHelper.get_score(logits)
            cam = ModuleHelper.cam(conv_out, score, self.linear_lists[0][0], self.reduction)
            return score, cam
        
        if extract_score:
            logits = self.linear_lists[0][0](x, None)
            score = ModuleHelper.get_score(logits)
            return score

        return out.flatten()


