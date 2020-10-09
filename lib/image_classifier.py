#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Class Definition for Image Classifier.


import os
import time

import torch
from lib.datasets.data_loader import DataLoader
from lib.models.tools.model_manager import ModelManager
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.helpers.runner_helper import RunnerHelper
from lib.utils.tools.average_meter import AverageMeter, DictAverageMeter
from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.running_score import RunningScore
from lib.utils.tools.trainer import Trainer


class ImageClassifier(object):
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

        self.train_loader = self.cls_data_loader.get_trainloader()
        self.val_loader = self.cls_data_loader.get_valloader()
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
        if self.configer.get('network', 'resume_val'):
            self.val()

        while self.runner_state['iters'] < self.solver_dict['max_iters']:
            self.train()

        self.val()

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.cls_net.train()
        start_time = time.time()
        # Adjust the learning rate after every epoch.
        self.runner_state['epoch'] += 1
        for i, data_dict in enumerate(self.train_loader):
            data_dict = {'src0_{}'.format(k): v for k, v in data_dict.items()}
            Trainer.update(self, warm_list=(0, ),
                           warm_lr_list=(self.solver_dict['lr']['base_lr']*self.configer.get('network', 'bb_lr_scale'),),
                           solver_dict=self.solver_dict)
            self.data_time.update(time.time() - start_time)
            data_dict = RunnerHelper.to_device(self, data_dict)
            # Forward pass.
            out = self.cls_net(data_dict)
            loss_dict, _ = self.loss(out)
            # Compute the loss of the train batch & backward.

            loss = loss_dict['loss']
            self.train_losses.update({key: loss.item() for key, loss in loss_dict.items()}, data_dict['src0_img'].size(0))
            self.optimizer.zero_grad()
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
                         'Learning rate = {4}\tLoss = {3}\n'.format(
                             self.runner_state['epoch'], self.runner_state['iters'],
                             self.solver_dict['display_iter'], self.train_losses.info(),
                             RunnerHelper.get_lr(self.optimizer), batch_time=self.batch_time,
                             data_time=self.data_time))

                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            if self.solver_dict['lr']['metric'] == 'iters' and self.runner_state['iters'] == self.solver_dict['max_iters']:
                break

            if self.runner_state['iters'] % self.solver_dict['save_iters'] == 0 and self.configer.get('local_rank') == 0:
                RunnerHelper.save_net(self, self.cls_net)

            # Check to val the current model.
            # if self.runner_state['iters'] % self.solver_dict['test_interval'] == 0 \
            #         and not self.configer.get('distributed'):
            #     self.val()
            if self.runner_state['iters'] % self.solver_dict['test_interval'] == 0:
                self.val()

    def val(self):
        """
          Validation function during the train phase.
        """
        self.cls_net.eval()
        start_time = time.time()
        with torch.no_grad():
            for j, data_dict in enumerate(self.val_loader):
                data_dict = {'src0_{}'.format(k): v for k, v in data_dict.items()}
                # Forward pass.
                data_dict = RunnerHelper.to_device(self, data_dict)
                out = self.cls_net(data_dict)
                loss_dict = self.loss(out)
                out_dict, label_dict, _ = RunnerHelper.gather(self, out)
                self.running_score.update(out_dict, label_dict)
                self.val_losses.update({key: loss.item() for key, loss in loss_dict.items()}, data_dict['src0_img'].size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            # RunnerHelper.save_net(self, self.cls_net) # only local_rank=0 can save net
            # Print the log info & reset the states.
            Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))
            Log.info('TestLoss = {}'.format(self.val_losses.info()))
            Log.info('Top1 ACC = {}'.format(self.running_score.get_top1_acc()))
            Log.info('Top3 ACC = {}'.format(self.running_score.get_top3_acc()))
            Log.info('Top5 ACC = {}'.format(self.running_score.get_top5_acc()))
            self.batch_time.reset()
            self.batch_time.reset()
            self.val_losses.reset()
            self.running_score.reset()
            self.cls_net.train()


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
