#!/usr/bin/env python
# -*- coding:utf-8 -*-


import torch.nn as nn

from lib.models.backbones.resnet.resnet_models import ResNetModels


class NormalResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()

        self.num_features = 2048
        # take pretrained resnet, except AvgPool and FC
        self.prefix = orig_resnet.prefix
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        x = self.prefix(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.resnet_models = ResNetModels(self.configer)

    def __call__(self, arch=None, pretrained_model=None, **kwargs):
        if arch == 'resnet18':
            orig_resnet = self.resnet_models.resnet18(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)
            arch_net.num_features = 512

        elif arch == 'resnet34':
            orig_resnet = self.resnet_models.resnet34(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)
            arch_net.num_features = 512

        elif arch == 'resnet50':
            orig_resnet = self.resnet_models.resnet50(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'resnet101':
            orig_resnet = self.resnet_models.resnet101(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'resnet152':
            orig_resnet = self.resnet_models.resnet152(pretrained=pretrained_model, **kwargs)
            arch_net = NormalResnetBackbone(orig_resnet)

        else:
            raise Exception('Architecture undefined!')

        return arch_net
