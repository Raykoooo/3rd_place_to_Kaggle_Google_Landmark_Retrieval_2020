#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Data loader for Image Classification.


import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing

from lib.utils.parallel.data_container import DataContainer
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log


class DefaultLoader(data.Dataset):

    def __init__(self, root_dir=None, dataset=None, label_path=None,
                 aug_transform=None, img_transform=None, configer=None, split_trainval=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        if split_trainval:
            self.img_list, self.label_list = self.__read_and_split_file(root_dir, dataset, label_path)
        else:
            self.img_list, self.label_list = self.__read_file(root_dir, dataset, label_path)

    def __getitem__(self, index):
        img = None
        valid = True
        while img is None:
            try:
                img = ImageHelper.read_image(self.img_list[index],
                                             tool=self.configer.get('data', 'image_tool'),
                                             mode=self.configer.get('data', 'input_mode'))
                assert isinstance(img, np.ndarray) or isinstance(img, Image.Image)
            except:
                Log.warn('Invalid image path: {}'.format(self.img_list[index]))
                img = None
                valid = False
                index = (index + 1) % len(self.img_list)

        label = torch.from_numpy(np.array(self.label_list[index]))
        if self.aug_transform is not None:
            img = self.aug_transform(img, self.label_list[index])

        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(
            valid=valid,
            img=DataContainer(img, stack=True),
            label=DataContainer(label, stack=True)
        )

    def __len__(self):

        return len(self.img_list)

    def __read_and_split_file(self, root_dir, dataset, label_path):
        img_list = list()
        mlabel_list = list()
        select_interval = int(1 / self.configer.get('data', 'val_ratio'))
        img_dict = dict()
        with open(label_path, 'r') as file_stream:
            for line in file_stream.readlines():
                label = line.strip().split()[1]
                if int(label) in img_dict:
                    img_dict[int(label)].append(line)
                else:
                    img_dict[int(label)] = [line]

        all_img_list = []
        for i in sorted(img_dict.keys()):
            all_img_list += img_dict[i]

        for line_cnt in range(len(all_img_list)):
            if line_cnt % select_interval == 0 and dataset == 'train' and not self.configer.get('data', 'include_val'):
                continue

            if line_cnt % select_interval != 0 and dataset == 'val':
                continue

            line_items = all_img_list[line_cnt].strip().split()
            path = line_items[0]
            if not os.path.exists(os.path.join(root_dir, path)) or not ImageHelper.is_img(path):
                Log.warn('Invalid Image Path: {}'.format(os.path.join(root_dir, path)))
                continue

            img_list.append(os.path.join(root_dir, path))
            mlabel_list.append([int(item) for item in line_items[1:]])

        assert len(img_list) > 0
        Log.info('Length of {} imgs is {} after split trainval...'.format(dataset, len(img_list)))
        return img_list, mlabel_list

    def __read_file(self, root_dir, dataset, label_path):
        img_list = list()
        mlabel_list = list()
        
        with open(label_path, 'r') as file_stream:
            for line in file_stream.readlines():
                line_items = line.rstrip().split()
                path = line_items[0]
                if not os.path.exists(os.path.join(root_dir, path)) or not ImageHelper.is_img(path):
                    Log.warn('Invalid Image Path: {}'.format(os.path.join(root_dir, path)))
                    continue

                img_list.append(os.path.join(root_dir, path))
                mlabel_list.append([int(item) for item in line_items[1:]])

        assert len(img_list) > 0
        Log.info('Length of {} imgs is {}...'.format(dataset, len(img_list)))
        return img_list, mlabel_list

if __name__ == "__main__":
    # Test data loader.
    pass
