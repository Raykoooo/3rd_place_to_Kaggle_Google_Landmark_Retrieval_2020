# -*- coding: utf-8 -*-

import torch

def spoc(feat):
    """
    SPoC with center prior.
    c.f. https://arxiv.org/pdf/1510.07493.pdf
    """
    b, c, h, w = feat.shape
    sigma = min(h, w) / 2.0 / 3.0
    x = torch.Tensor(range(w))
    y = torch.Tensor(range(h))[:, None]
    spatial_weight = torch.exp(-((x - (w - 1) / 2.0) ** 2 + (y - (h - 1) / 2.0) ** 2) / 2.0 / (sigma ** 2))
    if torch.cuda.is_available():
        spatial_weight = spatial_weight.to(feat.device)
    spatial_weight = spatial_weight[None, None, :, :]
    feat = (feat * spatial_weight).sum(dim=(2, 3))
    feat = feat.reshape((b, c, 1, 1))
    return feat
