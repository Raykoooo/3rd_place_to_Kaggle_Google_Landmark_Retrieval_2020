# -*- coding: utf-8 -*-

import queue

import torch

def bfs(x, y, mask, cc_map, cc_id):
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    q = queue.LifoQueue()
    q.put((x, y))

    ret = 1
    cc_map[x][y] = cc_id

    while not q.empty():
        x, y = q.get()

        for (dx, dy) in dirs:
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < mask.shape[0] and 0 <= new_y < mask.shape[1]:
                if mask[new_x][new_y] == 1 and cc_map[new_x][new_y] == 0:
                    q.put((new_x, new_y))
                    ret += 1
                    cc_map[new_x][new_y] = cc_id
    return ret

def find_max_cc(mask):
    """
    Find the largest connected component of the mask。

    Args:
        mask (torch.tensor): the original mask.

    Returns:
        mask (torch.tensor): the mask only containing the maximum connected component.
    """
    assert mask.ndimension() == 4
    assert mask.shape[1] == 1
    mask = mask[:, 0, :, :]
    for i in range(mask.shape[0]):
        m = mask[i]
        cc_map = torch.zeros(m.shape)
        cc_num = list()

        for x in range(m.shape[0]):
            for y in range(m.shape[1]):
                if m[x][y] == 1 and cc_map[x][y] == 0:
                    cc_id = len(cc_num) + 1
                    cc_num.append(bfs(x, y, m, cc_map, cc_id))

        max_cc_id = cc_num.index(max(cc_num)) + 1
        m[cc_map != max_cc_id] = 0
    mask = mask[:, None, :, :]
    return mask

def scda(feat):
    """
    Selective Convolutional Descriptor Aggregation for Fine-Grained Image Retrieval.
    c.f. http://www.weixiushen.com/publication/tip17SCDA.pdf
    """
    b, c, h, w = feat.shape
    mask = feat.sum(dim=1, keepdim=True)
    thres = mask.mean(dim=(2, 3), keepdim=True)
    mask[mask <= thres] = 0
    mask[mask > thres] = 1
    mask = find_max_cc(mask)
    feat = feat * mask

    gap = feat.mean(dim=(2, 3))
    gmp, _ = feat.max(dim=3)
    gmp, _ = gmp.max(dim=2)

    feat = torch.cat([gap, gmp], dim=1)
    feat = feat.reshape((b, c*2, 1, 1))
    return feat
