#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ResNet in PyTorch.


import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.loss.loss import BASE_LOSS_DICT
from lib.models.tools.metric_linear import LpNormalize
from lib.models.tools.module_helper import ModuleHelper

LOSS_TYPE = {
    'ce_loss': {
        'ce_src0_label0': 1.0, 'ce_src0_label1': 1.0,
        'ce_src1_label0': 1.0,
    },
    'mixup_ce_loss': {
        'ce_src0_label0': 1.0,
        'ce_src1_label0': 1.0,
        'mixup_ce_src1_label0': 1.0,
    },
    'sfl_loss': {
        'sfl_src0_label0': 1.0,'sfl_src0_label1': 1.0,'sfl_src1_label0': 1.0,
        'sfl_global_label': 1.0,
    },
    'triplet_margin_ce_loss': {
        'ce_src0_label0': 1.0,
        'triplet_margin_src0_label0': 1.0,
    },
    'online_triplet_margin_ce_loss': {
        'ce_src0_label0': 1.0,
        'online_triplet_margin_src1_label0': 1.0,
    },
    'multi_similarity_ce_loss': {
        'ce_src0_label0': 1.0,
        'multi_similarity_src0_label0': 0.1,
    },
    'circle_ce_loss': {
        'ce_src0_label0': 1.0,
        'circle_src0_label0': 0.1,
    },
    'smooth_ap_ce_loss': {
        'ce_src0_label0': 1.0,
        'smooth_ap_src0_label0': 1.0,
    },
    'smooth_ap_loss': {
        'smooth_ap_src0_label0': 1.0,
        'smooth_ap_src1_label0': 1.0,
    },
    'triplet_margin_loss': {
        'triplet_margin_src0_label0': 1.0,
        'triplet_margin_src1_label0': 1.0,
    },
}


class ClsModel(nn.Module):
    def __init__(self, configer, loss_dict=None, flag=""):
        super(ClsModel, self).__init__()
        self.configer = configer
        self.flag = flag if len(flag) == 0 else "{}_".format(flag)
        self.backbone = BackboneSelector(self.configer).get_backbone(
            backbone_type=self.configer.get('network.{}backbone'.format(self.flag)),
            pretrained_model=self.configer.get('network.{}pretrained'.format(self.flag)),
            rm_last_stride=self.configer.get('network.{}rm_last_stride'.format(self.flag), default=False)
        )

        self.reduction = None
        fc_dim_out = self.configer.get('network.{}fc_dim'.format(self.flag), default=None)
        fc_dim = self.backbone.num_features
        if fc_dim_out is not None:
            self.reduction = nn.Conv2d(self.backbone.num_features, fc_dim_out, 1)
            fc_dim = fc_dim_out
        self.bn = None
        if self.configer.get('network.{}fc_bn'.format(self.flag), default=True):
            self.bn = nn.BatchNorm1d(fc_dim)
            nn.init.zeros_(self.bn.bias)
            self.bn.bias.requires_grad = False
        self.relu = None
        if self.configer.get('network.{}fc_relu'.format(self.flag), default=True):
            self.relu = nn.ReLU()

        self.linear_lists = nn.ModuleList()
        for source in range(self.configer.get('data', 'num_data_sources')):
            linear_list = nn.ModuleList()
            linear_type = self.configer.get('network', '{}src{}_linear_type'.format(self.flag, source))
            for num_classes in self.configer.get('data.src{}_num_classes'.format(source)):
                linear_list.append(ModuleHelper.Linear(linear_type)(fc_dim, num_classes))
            self.linear_lists.append(linear_list)

        self.global_linear = None
        if self.configer.get('data.global_num_classes', default=None) is not None:
            global_linear_type = self.configer.get('network', '{}global_linear_type'.format(self.flag))
            self.global_linear = ModuleHelper.Linear(global_linear_type)(fc_dim, self.configer.get('data.global_num_classes', default=None))

        self.embed_after_norm = self.configer.get('network.embed_after_norm', default=True)
        self.embed = None
        if self.configer.get('network.{}embed'.format(self.flag), default=True):
            feat_dim = self.configer.get('network', '{}feat_dim'.format(self.flag))
            embed = []
            embed.append(nn.Linear(fc_dim, feat_dim))
            if self.configer.get('network.{}embed_norm_type'.format(self.flag)) == 'L2':
                embed.append(LpNormalize(p = 2, dim = 1))
            elif self.configer.get('network.{}embed_norm_type'.format(self.flag)) == 'BN':
                embed.append(nn.BatchNorm1d(feat_dim))
            self.embed = nn.Sequential(*embed)

        self.valid_loss_dict = LOSS_TYPE[self.configer.get('loss', 'loss_type')] if loss_dict is None else loss_dict

    def forward(self, data_dict):
        out_dict = dict()
        label_dict = dict()
        loss_dict = dict()
        in_img = ModuleHelper.concat(data_dict, 'img')
        in_img = ModuleHelper.preprocess(in_img, self.configer.get('data.normalize'))
        x_ = self.backbone(in_img)
        x_ = F.adaptive_avg_pool2d(x_, 1)
        x_ = self.reduction(x_) if self.reduction else x_
        x_ = x_.view(x_.size(0), -1)
        fc_ = self.bn(x_) if self.bn else x_
        fc_ = self.relu(fc_) if self.relu else fc_

        # metric learning for each data source separately
        if self.embed:
            if self.embed_after_norm:
                feat_ = self.embed(fc_)
            else:
                feat_ = self.embed(x_)
        else:
            if self.embed_after_norm:
                feat_ = fc_
            else:
                feat_ = x_
        start_point = 0
        for source in range(self.configer.get('data', 'num_data_sources')):
            fc = fc_[start_point:start_point+data_dict['src{}_img'.format(source)].size(0)]
            feat = feat_[start_point:start_point+data_dict['src{}_img'.format(source)].size(0)]
            for i in range(len(self.linear_lists[source])):
                gt_label = data_dict['src{}_label'.format(source)][:, i]
                sub_out = self.linear_lists[source][i](fc, gt_label)
                out_dict['{}src{}_label{}'.format(self.flag, source, i)] = sub_out
                label_dict['{}src{}_label{}'.format(self.flag, source, i)] = gt_label
                if 'ce_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}ce_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[sub_out, gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_src{}_label{}'.format(source, i)]])
                    )
                if self.training and 'mixup_ce_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    mixup_gt_label = data_dict['src0_label'][:, i]
                    loss_dict['{}mixup_ce_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[sub_out, mixup_gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['mixup_ce_src{}_label{}'.format(source, i)]])
                    )
                if 'sfl_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}sfl_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[sub_out, gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['softmax_focal_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['sfl_src{}_label{}'.format(source, i)]])
                    )
                if 'tri_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}tri_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[feat, gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['hard_triplet_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['tri_src{}_label{}'.format(source, i)]])
                    )
                if 'ls_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}ls_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[feat, gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['lifted_structure_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['ls_src{}_label{}'.format(source, i)]])
                    )
                if 'triplet_margin_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}triplet_margin_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[feat, gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['triplet_margin_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['triplet_margin_src{}_label{}'.format(source, i)]])
                    )
                if 'online_triplet_margin_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}online_triplet_margin_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[feat, data_dict['src{}_label'.format(source)]],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['online_triplet_margin_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['online_triplet_margin_src{}_label{}'.format(source, i)]])
                    )
                if 'multi_similarity_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}multi_similarity_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[feat, gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['multi_similarity_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['multi_similarity_src{}_label{}'.format(source, i)]])
                    )
                if 'circle_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}circle_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[feat, gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['circle_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['circle_src{}_label{}'.format(source, i)]])
                    )
                if self.training and 'smooth_ap_src{}_label{}'.format(source, i) in self.valid_loss_dict:
                    loss_dict['{}smooth_ap_src{}_label{}'.format(self.flag, source, i)] = dict(
                        params=[feat, gt_label],
                        type=torch.cuda.LongTensor([BASE_LOSS_DICT['smooth_ap_loss']]),
                        weight=torch.cuda.FloatTensor([self.valid_loss_dict['smooth_ap_src{}_label{}'.format(source, i)]])
                    )

            start_point += data_dict['src{}_img'.format(source)].size(0)

        # metric learning for all data source together
        if self.global_linear:
            gt_label = ModuleHelper.concat(data_dict, 'label')[:, -1]
            sub_out = self.global_linear(fc_, gt_label)
            out_dict['{}global_label'.format(self.flag)] = sub_out
            label_dict['{}global_label'.format(self.flag)] = gt_label
            if 'ce_global_label' in self.valid_loss_dict:
                loss_dict['{}ce_global_label'.format(self.flag)] = dict(
                    params=[sub_out, gt_label],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_global_label']])
                )
            if 'sfl_global_label' in self.valid_loss_dict:
                loss_dict['{}sfl_global_label'.format(self.flag)] = dict(
                    params=[sub_out, gt_label],
                    type=torch.cuda.LongTensor([BASE_LOSS_DICT['softmax_focal_loss']]),
                    weight=torch.cuda.FloatTensor([self.valid_loss_dict['sfl_global_label']])
                )


        return out_dict, label_dict, loss_dict
