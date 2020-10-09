#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import numpy as np
import torch.utils.data as data
from PIL import Image

from lib.utils.parallel.data_container import DataContainer
from lib.utils.helpers.file_helper import FileHelper
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log


class TestDefaultLoader(data.Dataset):

    def __init__(self, test_dir=None, aug_transform=None, img_transform=None, configer=None):
        super(TestDefaultLoader, self).__init__()
        self.configer = configer
        self.aug_transform=aug_transform
        self.img_transform = img_transform
        self.item_list = [(os.path.abspath(os.path.join(test_dir, filename)), filename)
                          for filename in FileHelper.list_dir(test_dir) if ImageHelper.is_img(filename)]

    def __getitem__(self, index):
        img = None
        valid = True
        while img is None:
            try:
                img = ImageHelper.read_image(self.item_list[index][0],
                                             tool=self.configer.get('data', 'image_tool'),
                                             mode=self.configer.get('data', 'input_mode'))
                assert isinstance(img, np.ndarray) or isinstance(img, Image.Image)
            except:
                Log.warn('Invalid image path: {}'.format(self.item_list[index][0]))
                img = None
                valid = False
                index = (index + 1) % len(self.item_list)

        if self.aug_transform is not None:
            img = self.aug_transform(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        meta = dict(
            valid=valid,
            img_path=self.item_list[index][0],
            filename=self.item_list[index][1]
        )
        return dict(
            img=DataContainer(img, stack=True),
            meta=DataContainer(meta, stack=False, cpu_only=True)
        )

    def __len__(self):

        return len(self.item_list)


class TestListLoader(data.Dataset):

    def __init__(self, root_dir=None, list_path=None, aug_transform=None, img_transform=None, configer=None):
        super(TestListLoader, self).__init__()
        self.configer = configer
        self.aug_transform=aug_transform
        self.img_transform = img_transform
        self.item_list = self.__read_list(root_dir, list_path)

    def __getitem__(self, index):
        img = None
        valid = True
        while img is None:
            try:
                img = ImageHelper.read_image(self.item_list[index][0],
                                             tool=self.configer.get('data', 'image_tool'),
                                             mode=self.configer.get('data', 'input_mode'))
                assert isinstance(img, np.ndarray) or isinstance(img, Image.Image)
            except:
                Log.warn('Invalid image path: {}'.format(self.item_list[index][0]))
                img = None
                valid = False
                index = (index + 1) % len(self.item_list)

        ori_img_size = ImageHelper.get_size(img)
        if self.aug_transform is not None:
            img = self.aug_transform(img)

        border_hw = ImageHelper.get_size(img)[::-1]
        if self.img_transform is not None:
            img = self.img_transform(img)

        meta = dict(
            valid=valid,
            ori_img_size=ori_img_size,
            border_hw=border_hw,
            img_path=self.item_list[index][0],
            filename=self.item_list[index][1],
            label=self.item_list[index][2]
        )
        return dict(
            img=DataContainer(img, stack=True),
            meta=DataContainer(meta, stack=False, cpu_only=True)
        )

    def __len__(self):

        return len(self.item_list)

    def __read_list(self, root_dir, list_path):
        item_list = []
        with open(list_path, 'r') as fr:
            for line in fr.readlines():
                filename = line.strip().split()[0]
                label = None if len(line.strip().split()) == 1 else line.strip().split()[1]
                img_path = os.path.join(root_dir, filename)
                if not os.path.exists(img_path) or not ImageHelper.is_img(img_path):
                    Log.error('Image Path: {} is Invalid.'.format(img_path))
                    exit(1)

                item_list.append((img_path, filename, label))

        Log.info('There are {} images..'.format(len(item_list)))
        return item_list
