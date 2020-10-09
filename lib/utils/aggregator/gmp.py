# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

def gmp(feat):
    """
    Global maximum pooling
    """
    feat = F.adaptive_max_pool2d(feat, 1)
    return feat
