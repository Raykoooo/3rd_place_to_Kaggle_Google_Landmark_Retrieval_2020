#!/usr/bin/env python
# -*- coding:utf-8 -*-


import torch.nn as nn

from lib.models.backbones.resnest.resnest_models import ResNeStModels


class NormalResneStBackbone(nn.Module):
    def __init__(self, orig_resnest):
        super(NormalResneStBackbone, self).__init__()

        self.num_features = 2048
        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnest.conv1
        self.bn1 = orig_resnest.bn1
        self.relu = orig_resnest.relu
        self.maxpool = orig_resnest.maxpool
        self.layer1 = orig_resnest.layer1
        self.layer2 = orig_resnest.layer2
        self.layer3 = orig_resnest.layer3
        self.layer4 = orig_resnest.layer4

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNeStBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.resnest_models = ResNeStModels(self.configer)

    def __call__(self, arch=None, pretrained_model=None, **kwargs):
        if arch == 'resnest50':
            orig_resnest = self.resnest_models.resnest50(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResneStBackbone(orig_resnest)

        elif arch == 'resnest101':
            orig_resnest = self.resnest_models.resnest101(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResneStBackbone(orig_resnest)

        elif arch == 'resnest200':
            orig_resnest = self.resnest_models.resnest200(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResneStBackbone(orig_resnest)

        elif arch == 'resnest269':
            orig_resnest = self.resnest_models.resnest269(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResneStBackbone(orig_resnest)

        else:
            raise Exception('Architecture undefined!')

        return arch_net
