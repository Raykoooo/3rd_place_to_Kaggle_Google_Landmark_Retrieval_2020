#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Adapted from https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/collate.py


import collections
import torch
from torch.utils.data.dataloader import default_collate
from torch._six import string_classes, int_classes

from lib.utils.parallel.data_container import DataContainer


def stack(batch, data_key=None):
    if isinstance(batch[0][data_key], DataContainer):
        if batch[0][data_key].stack:
            assert isinstance(batch[0][data_key].data, torch.Tensor) or \
                   isinstance(batch[0][data_key].data, int_classes) or \
                   isinstance(batch[0][data_key].data, float) or \
                   isinstance(batch[0][data_key].data, string_classes) or \
                   isinstance(batch[0][data_key].data, collections.Mapping) or\
                   isinstance(batch[0][data_key].data, collections.Sequence)
            stacked = default_collate([sample[data_key].data for sample in batch])
            return stacked
        else:
            return [sample[data_key].data for sample in batch]

    else:
        return default_collate([sample[data_key] for sample in batch])


def collate(batch):
    data_keys = batch[0].keys()

    return dict({key: stack(batch, data_key=key) for key in data_keys})
