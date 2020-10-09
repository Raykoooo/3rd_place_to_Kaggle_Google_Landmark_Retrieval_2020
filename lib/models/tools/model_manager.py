#!/usr/bin/env python
# -*- coding:utf-8 -*-

from lib.models.nets.cls_model import ClsModel
from lib.models.nets.deploy_cls_model import DeployClsModel
from lib.models.loss.loss import Loss
from lib.utils.tools.logger import Logger as Log


CLS_MODEL_DICT = {
    'cls_model': ClsModel,
}

DEPLOY_MODEL_DICT = {
    'cls_model': DeployClsModel,
}


class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def get_model(self, model_type=None):
        model_name = self.configer.get('network', 'model_name') if model_type is None else model_type

        if model_name not in CLS_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = CLS_MODEL_DICT[model_name](self.configer)

        return model

    def get_deploy_model(self, model_type=None):
        model_name = self.configer.get('network', 'model_name') if model_type is None else model_type

        if model_name not in DEPLOY_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = DEPLOY_MODEL_DICT[model_name](self.configer)

        return model

    def get_loss(self):
        if self.configer.get('network', 'gather'):
            return Loss(self.configer)

        from lib.utils.parallel.data_parallel import DataParallelCriterion
        return DataParallelCriterion(Loss(self.configer))
