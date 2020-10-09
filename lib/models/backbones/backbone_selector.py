#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Select Cls Model for pose detection.


from lib.models.backbones.resnet.resnet_backbone import ResNetBackbone
from lib.models.backbones.resnest.resnest_backbone import ResNeStBackbone
from lib.utils.tools.logger import Logger as Log


BACKBONE_DICT = {
    'resnest': ResNeStBackbone,
    'resnet': ResNetBackbone,
}

class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, backbone_type=None, pretrained_model=None, **kwargs):
        backbone_type = self.configer.get('network', 'backbone') if backbone_type is None else backbone_type
        pretrained_model = self.configer.get('network', 'pretrained') if pretrained_model is None else pretrained_model
        for k, b in BACKBONE_DICT.items():
            if backbone_type.startswith(k):
                return BACKBONE_DICT[k](self.configer)(arch=backbone_type, pretrained_model=pretrained_model, **kwargs)

        raise Exception('Backbone {} is invalid!!!'.format(backbone_type))
