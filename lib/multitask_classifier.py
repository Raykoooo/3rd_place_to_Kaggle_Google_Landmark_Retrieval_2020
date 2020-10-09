#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Class Definition for Image Classifier.


import os
import time
import random
import torch
import torch.nn.functional as F
from lib.datasets.data_loader import DataLoader
from lib.models.tools.model_manager import ModelManager
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.helpers.runner_helper import RunnerHelper
from lib.utils.tools.average_meter import AverageMeter, DictAverageMeter
from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.running_score import RunningScore
from lib.utils.tools.trainer import Trainer
from apex import amp
import yaml

class MultiTaskClassifier(object):
    """
      The class for the training phase of Image classification.
    """
    def __init__(self, configer):
        self.configer = configer
        self.runner_state = dict(iters=0, last_iters=0, epoch=0,
                                 last_epoch=0, performance=0,
                                 val_loss=0, max_performance=0, min_val_loss=0)

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = DictAverageMeter()
        self.val_losses = DictAverageMeter()
        self.cls_model_manager = ModelManager(configer)
        self.cls_data_loader = DataLoader(configer)
        self.running_score = RunningScore(configer)

        self.cls_net = self.cls_model_manager.get_model()
        self.solver_dict = self.configer.get(self.configer.get('train', 'solver'))
        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.solver_dict)
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)
        self.cls_net, self.optimizer = RunnerHelper.to_dtype(self, self.cls_net, self.optimizer)

        self.train_loaders = dict()
        self.val_loaders = dict()
        for source in range(self.configer.get('data', 'num_data_sources')):
            self.train_loaders[source] = self.cls_data_loader.get_trainloader(source=source)
            self.val_loaders[source] = self.cls_data_loader.get_valloader(source=source)
        if self.configer.get('data', 'mixup'):
            assert(self.configer.get('data', 'num_data_sources') == 2), "mixup only support src0 and src1 load the same dataset"

        self.loss = self.cls_model_manager.get_loss()

    def _get_parameters(self):
        lr_1 = []
        lr_2 = []
        params_dict = dict(self.cls_net.named_parameters())
        for key, value in params_dict.items():
            if value.requires_grad:
                if 'backbone' in key:
                    if self.configer.get('network', 'bb_lr_scale') == 0.0:
                        value.requires_grad = False
                    else:
                        lr_1.append(value)
                else:
                    lr_2.append(value)

        params = [{'params': lr_1, 'lr': self.solver_dict['lr']['base_lr']*self.configer.get('network', 'bb_lr_scale')},
                  {'params': lr_2, 'lr': self.solver_dict['lr']['base_lr']}]
        return params

    def run(self):
        """
          Train function of every epoch during train phase.
        """
        if self.configer.get('network', 'resume_val'):
            self.val()

        self.cls_net.train()
        train_loaders = dict()
        for source in self.train_loaders:
            train_loaders[source] = iter(self.train_loaders[source])
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        while self.runner_state['iters'] < self.solver_dict['max_iters']:
            data_dict = dict()
            for source in train_loaders:
                try:
                    tmp_data_dict = next(train_loaders[source])
                    # Log.info('iter={}, source={}'.format(self.runner_state['iters'], source))
                except StopIteration:
                    if source == 0 or source == '0':
                        self.runner_state['epoch'] += 1
                    # Log.info('Repeat: iter={}, source={}'.format(self.runner_state['iters'], source))
                    train_loaders[source] = iter(self.train_loaders[source])
                    tmp_data_dict = next(train_loaders[source])
                for k, v in tmp_data_dict.items():
                    data_dict['src{}_{}'.format(source, k)] = v
            
            if self.configer.get('data', 'multiscale') is not None:
                scale_ratios = self.configer.get('data', 'multiscale')
                scale_ratio = random.uniform(scale_ratios[0], scale_ratios[-1])
                for key in data_dict:
                    if key.endswith('_img'):
                        data_dict[key] = F.interpolate(data_dict[key], scale_factor=[scale_ratio, scale_ratio], 
                                                    mode='bilinear', align_corners=True)
            if self.configer.get('data', 'mixup'):
                src0_resize = F.interpolate(data_dict['src0_img'], scale_factor=[random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)], 
                                            mode='bilinear', align_corners=True)
                b, c, h, w = src0_resize.shape
                pos = random.randint(0, 3)
                if pos == 0:  # top-left
                    data_dict['src1_img'][:, :, 0:h, 0:w] = src0_resize
                elif pos == 1:  # top-right
                    data_dict['src1_img'][:, :, 0:h, -w:] = src0_resize
                elif pos == 2:  # bottom-left
                    data_dict['src1_img'][:, :, -h:, 0:w] = src0_resize
                else:  # bottom-right
                    data_dict['src1_img'][:, :, -h:, -w:] = src0_resize
                
            data_dict = RunnerHelper.to_device(self, data_dict)
            Trainer.update(self, warm_list=(0, ),
                           warm_lr_list=(self.solver_dict['lr']['base_lr']*self.configer.get('network', 'bb_lr_scale'),),
                           solver_dict=self.solver_dict)
            self.data_time.update(time.time() - start_time)
            # Forward pass.
            out = self.cls_net(data_dict)
            loss_dict, loss_weight_dict = self.loss(out)
            # Compute the loss of the train batch & backward.
            loss = loss_dict['loss']
            self.train_losses.update({key: loss.item() for key, loss in loss_dict.items()}, data_dict['src0_img'].size(0))
            self.optimizer.zero_grad()
            if self.configer.get('dtype') == 'fp16':
                with amp.scale_loss(loss, self.optimizer) as scaled_losses:
                    scaled_losses.backward() 
            else:
                loss.backward()
            if self.configer.get('network', 'clip_grad'):
                RunnerHelper.clip_grad(self.cls_net, 10.)

            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.runner_state['iters'] += 1

            # Print the log info & reset the states.
            if self.runner_state['iters'] % self.solver_dict['display_iter'] == 0:
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {5}\tLoss = {3}\nLossWeight = {4}\n'.format(
                             self.runner_state['epoch'], self.runner_state['iters'],
                             self.solver_dict['display_iter'], self.train_losses.info(),
                             loss_weight_dict,
                             RunnerHelper.get_lr(self.optimizer), batch_time=self.batch_time,
                             data_time=self.data_time))

                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            if self.solver_dict['lr']['metric'] == 'iters' and self.runner_state['iters'] == self.solver_dict['max_iters']:
                if self.configer.get('local_rank') == 0:
                    RunnerHelper.save_net(self, self.cls_net, postfix='final')
                break

            if self.runner_state['iters'] % self.solver_dict['save_iters'] == 0 and self.configer.get('local_rank') == 0:
                RunnerHelper.save_net(self, self.cls_net)

            # Check to val the current model.
            if self.runner_state['iters'] % self.solver_dict['test_interval'] == 0:
                self.val()
                if self.configer.get('local_rank') == 0:
                    RunnerHelper.save_net(self, self.cls_net, performance=self.runner_state['performance'])

        self.val()

    def val(self):
        """
          Validation function during the train phase.
        """
        self.cls_net.eval()
        start_time = time.time()
        val_loaders = dict()
        val_to_end = dict()
        all_to_end = False
        for source in self.val_loaders:
            val_loaders[source] = iter(self.val_loaders[source])
            val_to_end[source] = False
        with torch.no_grad():
            while not all_to_end:
                data_dict = dict()
                for source in val_loaders:
                    try:
                        tmp_data_dict = next(val_loaders[source])
                    except StopIteration:
                        val_to_end[source] = True
                        val_loaders[source] = iter(self.val_loaders[source])
                        tmp_data_dict = next(val_loaders[source])
                    for k, v in tmp_data_dict.items():
                        data_dict['src{}_{}'.format(source, k)] = v
                # Forward pass.
                data_dict = RunnerHelper.to_device(self, data_dict)
                out = self.cls_net(data_dict)
                loss_dict, loss_weight_dict = self.loss(out)
                out_dict, label_dict, _ = RunnerHelper.gather(self, out)
                # Compute the loss of the val batch.
                self.running_score.update(out_dict, label_dict)
                self.val_losses.update({key: loss.mean().item() for key, loss in loss_dict.items()}, data_dict['src0_img'].size(0))
                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()
                # check whether scan over all data sources
                all_to_end = True
                for source in val_to_end:
                    if not val_to_end[source]:
                        all_to_end = False

            Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))
            Log.info('TestLoss = {}'.format(self.val_losses.info()))
            Log.info('TestLossWeight = {}'.format(loss_weight_dict))
            Log.info('Top1 ACC = {}'.format(self.running_score.get_top1_acc()))
            Log.info('Top3 ACC = {}'.format(self.running_score.get_top3_acc()))
            Log.info('Top5 ACC = {}'.format(self.running_score.get_top5_acc()))
            top1_acc = yaml.load(self.running_score.get_top1_acc())
            for key in top1_acc:
                if 'src0_label0' in key:
                    self.runner_state['performance'] = top1_acc[key]
                    Log.info('Use acc of {} to compare performace'.format(key))
                    break
            self.running_score.reset()
            self.val_losses.reset()
            self.batch_time.reset()
            self.cls_net.train()

if __name__ == "__main__":
    # Test class for pose estimator.
    pass
