# -*- coding: utf-8 -*-

import torch

def crow(feat):
    """
    Cross-dimensional Weighting for Aggregated Deep Convolutional Features.
    c.f. https://arxiv.org/pdf/1512.04065.pdf

    Hyper-Params
        spatial_a (float): hyper-parameter for calculating spatial weight.
        spatial_b (float): hyper-parameter for calculating spatial weight.
    """
    default_hyper_params = {
        "spatial_a": 2.0,
        "spatial_b": 2.0,
    }

    spatial_a = default_hyper_params["spatial_a"]
    spatial_b = default_hyper_params["spatial_b"]

    spatial_weight = feat.sum(dim=1, keepdim=True)
    z = (spatial_weight ** spatial_a).sum(dim=(2, 3), keepdim=True)
    z = z ** (1.0 / spatial_a)
    spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)

    b, c, w, h = feat.shape
    nonzeros = (feat!=0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
    channel_weight = torch.log(nonzeros.sum(dim=1, keepdim=True) / nonzeros)

    feat = feat * spatial_weight
    feat = feat.sum(dim=(2, 3))
    feat = feat * channel_weight
    feat = feat.reshape((b, c, 1, 1))

    return feat
