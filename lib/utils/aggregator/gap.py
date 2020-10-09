# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

def gap(feat):
    feat = F.adaptive_avg_pool2d(feat, 1)
    return feat
