# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

def rmac_simple(feat):
    feat = F.max_pool2d(feat, kernel_size=(7, 7), stride=2, padding=3)
    feat = F.adaptive_avg_pool2d(feat, 1)
    return feat
