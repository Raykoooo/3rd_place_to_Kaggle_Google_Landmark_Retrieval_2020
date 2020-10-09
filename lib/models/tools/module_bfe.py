#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random
import torch.nn as nn
import torch.nn.functional as F
import torch


class BatchDrop(nn.Module):
    def __init__(self, configer):
        super(BatchDrop, self).__init__()
        self.h_ratio = configer.get('bfe', 'height_ratio')
        self.w_ratio = configer.get('bfe', 'width_ratio')

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x


class BatchCrop(nn.Module):
    def __init__(self, configer):
        super(BatchCrop, self).__init__()
        self.ratio = configer.get('bfe', 'height_ratio')

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = int(self.ratio * h)
            start = random.randint(0, h-1)
            if start + rh > h:
                select = list(range(0, start+rh-h)) + list(range(start, h))
            else:
                select = list(range(start, start+rh))
            mask = x.new_zeros(x.size())
            mask[:, :, select, :] = 1
            x = x * mask
        return x


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, configer):
        super(DropBlock2D, self).__init__()

        self.drop_prob = configer.get('bfe', 'drop_prob')
        self.block_size = configer.get('bfe', 'drop_block_size')

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            assert x.dim() == 4, \
                "Expected input with 4 dimensions (bsize, channels, height, width)"
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    def __init__(self, configer):
        super(LinearScheduler, self).__init__()
        self.dropblock = DropBlock2D(configer)
        self.start_value = configer.get('bfe', 'drop_prob_start')
        self.stop_value = configer.get('bfe', 'drop_prob')
        self.nsteps = configer.get('bfe', 'drop_prob_nsteps')
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nsteps)

    def forward(self, x):
        self.step()
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1
        