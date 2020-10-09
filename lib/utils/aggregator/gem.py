# -*- coding: utf-8 -*-

import torch

def gem(feat):
    """
    Generalized-mean pooling.
    c.f. https://pdfs.semanticscholar.org/a2ca/e0ed91d8a3298b3209fc7ea0a4248b914386.pdf

    Hyper-Params
        p (float): hyper-parameter for calculating generalized mean. If p = 1, GeM is equal to global average pooling, and
            if p = +infinity, GeM is equal to global max pooling.
    """
    default_hyper_params = {
        "p": 3.0,
    }

    p = default_hyper_params["p"]

    feat = feat ** p
    b, c, h, w = feat.shape
    feat = feat.sum(dim=(2, 3)) * 1.0 / w / h
    feat = feat ** (1.0 / p)
    feat = feat.reshape((b, c, 1, 1))
    return feat
