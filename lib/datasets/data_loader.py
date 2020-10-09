#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Class for the Pose Data Loader.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils import data

import lib.datasets.tools.pil_aug_transforms as pil_aug_trans
import lib.datasets.tools.cv2_aug_transforms as cv2_aug_trans
import lib.datasets.tools.transforms as trans
from lib.datasets.tools.collate import collate
from lib.datasets.tools.sampler import RankingSampler, BalanceSampler, ReverseSampler, OnlineTripletSampler
from lib.datasets.loader.default_loader import DefaultLoader
from lib.datasets.loader.test_loader import TestDefaultLoader, TestListLoader
from lib.utils.tools.logger import Logger as Log


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_val_transform = pil_aug_trans.PILAugCompose(self.configer, split='val')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='val')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_test_transform = pil_aug_trans.PILAugCompose(self.configer, split='test')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_test_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='test')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
        ])

    def get_trainloader(self, source=0, loader_type=None, root_dir=None, label_path=None,
                        batch_size=None, samples_per_class=None, min_cnt=None, max_cnt=None):
        loader_type = self.configer.get('train', 'src{}_loader'.format(source)) if loader_type is None else loader_type
        label_path = self.configer.get('data', 'src{}_label_path'.format(source)) if label_path is None else label_path
        root_dir = self.configer.get('data', 'src{}_data_dir'.format(source)) if root_dir is None else root_dir
        batch_size = self.configer.get('train', 'src{}_batch_size'.format(source)) if batch_size is None else batch_size
        min_cnt = self.configer.get('train', 'src{}_min_count'.format(source)) if min_cnt is None else min_cnt
        max_cnt = self.configer.get('train', 'src{}_max_count'.format(source)) if max_cnt is None else max_cnt
        samples_per_class = self.configer.get('train', 'src{}_samples_per_class'.format(source)) if samples_per_class is None else samples_per_class
        split_trainval = False if self.configer.get('data', 'src{}_val_label_path'.format(source)) is not None else True
        
        if loader_type is None or loader_type == 'default':
            dataset = DefaultLoader(root_dir=root_dir, dataset='train', label_path=label_path,
                                    aug_transform=self.aug_train_transform, img_transform=self.img_transform, 
                                    configer=self.configer, split_trainval=split_trainval)
            trainloader = data.DataLoader(
                dataset=dataset,
                batch_sampler=BalanceSampler(
                    label_list=dataset.label_list, batch_size=batch_size, min_cnt=min_cnt, max_cnt=max_cnt,
                    is_distributed=self.configer.get('distributed', default=False)
                ),
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return trainloader

        elif loader_type == 'ranking':
            dataset = DefaultLoader(root_dir=root_dir, dataset='train', label_path=label_path,
                                    aug_transform=self.aug_train_transform, img_transform=self.img_transform, 
                                    configer=self.configer, split_trainval=split_trainval)
            trainloader = data.DataLoader(
                dataset=dataset,
                batch_sampler=RankingSampler(
                    label_list=dataset.label_list, samples_per_class=samples_per_class,
                    batch_size=batch_size, min_cnt=min_cnt, max_cnt=max_cnt,
                    is_distributed=self.configer.get('distributed', default=False)
                ),
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return trainloader

        elif loader_type == 'reverse':
            dataset = DefaultLoader(root_dir=root_dir, dataset='train', label_path=label_path,
                                    aug_transform=self.aug_train_transform, img_transform=self.img_transform, 
                                    configer=self.configer, split_trainval=split_trainval)
            trainloader = data.DataLoader(
                dataset=dataset,
                batch_sampler=ReverseSampler(
                    label_list=dataset.label_list, batch_size=batch_size, min_cnt=min_cnt, max_cnt=max_cnt,
                    is_distributed=self.configer.get('distributed', default=False)
                ),
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return trainloader

        elif loader_type == 'online_triplet':
            dataset = DefaultLoader(root_dir=root_dir, dataset='train', label_path=label_path,
                                    aug_transform=self.aug_train_transform, img_transform=self.img_transform, 
                                    configer=self.configer, split_trainval=split_trainval)
            trainloader = data.DataLoader(
                dataset=dataset,
                batch_sampler=OnlineTripletSampler(
                    label_list=dataset.label_list, batch_size=batch_size, min_cnt=min_cnt, max_cnt=max_cnt,
                    is_distributed=self.configer.get('distributed', default=False)
                ),
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return trainloader           

        else:
            Log.error('{} train loader is invalid.'.format(self.configer.get('train', 'loader')))
            exit(1)

    def get_valloader(self, source=0, loader_type=None, root_dir=None, label_path=None, batch_size=None):
        loader_type = self.configer.get('val', 'src{}_loader'.format(source)) if loader_type is None else loader_type
        if label_path is None:
            if self.configer.get('data', 'src{}_val_label_path'.format(source)) is not None:
                label_path = self.configer.get('data', 'src{}_val_label_path'.format(source))
                split_trainval = False
            else:
                label_path = self.configer.get('data', 'src{}_label_path'.format(source))
                split_trainval = True
        if root_dir is None:
            if self.configer.get('data', 'src{}_val_data_dir'.format(source)) is not None:
                root_dir = self.configer.get('data', 'src{}_val_data_dir'.format(source))
            else:
                root_dir = self.configer.get('data', 'src{}_data_dir'.format(source))
        batch_size = self.configer.get('val', 'src{}_batch_size'.format(source)) if batch_size is None else batch_size
        if loader_type is None or loader_type == 'default':
            valloader = data.DataLoader(
                DefaultLoader(root_dir=root_dir, dataset='val', label_path=label_path,
                              aug_transform=self.aug_val_transform, img_transform=self.img_transform, 
                              configer=self.configer, split_trainval=split_trainval),
                batch_size=batch_size, shuffle=False,
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return valloader

        elif loader_type == 'online_triplet':
            dataset = DefaultLoader(root_dir=root_dir, dataset='val', label_path=label_path,
                                    aug_transform=self.aug_val_transform, img_transform=self.img_transform, 
                                    configer=self.configer, split_trainval=split_trainval)
            valloader = data.DataLoader(
                dataset=dataset,
                batch_sampler=OnlineTripletSampler(
                    label_list=dataset.label_list, batch_size=batch_size, min_cnt=0, max_cnt=-1,
                    is_distributed=self.configer.get('distributed', default=False)
                ),
                num_workers=self.configer.get('data', 'workers'), pin_memory=True, collate_fn=collate
            )

            return valloader

        else:
            Log.error('{} val loader is invalid.'.format(self.configer.get('val', 'loader')))
            exit(1)

    def get_testloader(self, test_dir=None, list_path=None, root_dir="/", batch_size=1, workers=8):
        if test_dir is not None:
            assert list_path is None
            testloader = data.DataLoader(
                TestDefaultLoader(test_dir=test_dir,
                                  aug_transform=self.aug_test_transform,
                                  img_transform=self.img_transform,
                                  configer=self.configer),
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True, collate_fn=collate
            )
            return testloader

        elif list_path is not None:
            testloader = data.DataLoader(
                TestListLoader(root_dir=root_dir,
                               list_path=list_path,
                               aug_transform=self.aug_test_transform,
                               img_transform=self.img_transform,
                               configer=self.configer),
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True, collate_fn=collate
            )
            return testloader

        else:
            Log.error('Params is invalid.')
            exit(1)


if __name__ == "__main__":
    # Test data loader.
    pass
