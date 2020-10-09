#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Class Definition for Image Classifier.


import os
import time
import torch
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from lib.datasets.data_loader import DataLoader
from lib.utils.helpers.runner_helper import RunnerHelper
from lib.utils.tools.trainer import Trainer
from lib.models.tools.model_manager import ModelManager
from lib.utils.tools.average_meter import AverageMeter
from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.running_score import RunningScore


class AutoSKUMerger(object):
    """
      The class for the training phase of Image classification.
    """
    def __init__(self, configer):
        self.configer = configer
        self.cls_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.runner_state = None
        self.round = 1

        self._relabel()

    def _relabel(self):
        label_id = 0
        label_dict = dict()
        old_label_path = self.configer.get('data', 'label_path')
        new_label_path = '{}_new'.format(self.configer.get('data', 'label_path'))
        self.configer.update('data.label_path', new_label_path)
        fw = open(new_label_path, 'w')
        check_valid_dict = dict()
        with open(old_label_path, 'r') as fr:
            for line in fr.readlines():
                line_items = line.strip().split()
                if not os.path.exists(os.path.join(self.configer.get('data', 'data_dir'), line_items[0])):
                    continue

                if line_items[1] not in label_dict:
                    label_dict[line_items[1]] = label_id
                    label_id += 1

                if line_items[0] in check_valid_dict:
                    Log.error('Duplicate Error: {}'.format(line_items[0]))
                    exit()

                check_valid_dict[line_items[0]] = 1
                fw.write('{} {}\n'.format(line_items[0], label_dict[line_items[1]]))

        fw.close()
        shutil.copy(self.configer.get('data', 'label_path'),
                    os.path.join(self.configer.get('data', 'merge_dir'), 'ori_label.txt'))
        self.configer.update(('data.num_classes'), [label_id])
        Log.info('Num Classes is {}...'.format(self.configer.get('data', 'num_classes')))

    def _init(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.cls_model_manager = ModelManager(self.configer)
        self.cls_data_loader = DataLoader(self.configer)
        self.cls_running_score = RunningScore(self.configer)
        self.runner_state = dict(iters=0, last_iters=0, epoch=0,
                                 last_epoch=0, performance=0,
                                 val_loss=0, max_performance=0, min_val_loss=0)
        self.cls_net = self.cls_model_manager.get_model()
        self.cls_net = RunnerHelper.load_net(self, self.cls_net)
        self.solver_dict = self.configer.get(self.configer.get('train', 'solver'))
        self.optimizer, self.scheduler = Trainer.init(self._get_parameters(), self.solver_dict)

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

    def _merge_class(self, cmatrix, fscore_list):
        Log.info('Merging class...')
        Log.info('Avg F1-score: {}'.format(fscore_list[-1]))
        threshold = max(self.configer.get('merge', 'min_thre'),
                        self.configer.get('merge', 'max_thre') - self.configer.get('merge', 'round_decay') * self.round)
        h, w = cmatrix.shape[0], cmatrix.shape[1]
        per_class_num = np.sum(cmatrix, 1)
        pairs_list = list()
        pre_dict = dict()
        for i in range(h):
            for j in range(w):
                if i == j:
                    continue

                if cmatrix[i][j] * 1.0 / per_class_num[i] > threshold:
                    pairs_list.append([i, j])
                    pre_dict[i] = i
                    pre_dict[j] = j

        for pair in pairs_list:
            root_node = list()
            for item in pair:
                r = item
                while pre_dict[r] != r:
                    r = pre_dict[r]

                i = item
                while i != r:
                    j = pre_dict[i]
                    pre_dict[i] = r
                    i = j

                root_node.append(r)

            if root_node[0] != root_node[1]:
                pre_dict[root_node[0]] = root_node[1]

        pairs_dict = dict()
        for k in pre_dict.keys():
            v = k
            while pre_dict[v] != v:
                v = pre_dict[v]

            if v != k:
                if v not in pairs_dict:
                    pairs_dict[v] = [k]
                else:
                    pairs_dict[v].append(k)

        mutual_pairs_dict = {}
        for k, v in pairs_dict.items():
            mutual_pairs_dict[k] = v
            if len(v) > 1:  # multi relation
                for p in v:
                    mutual_pairs_dict[p] = [k]
                    for q in v:
                        if p != q:
                            mutual_pairs_dict[p].append(q)

            else:
                mutual_pairs_dict[v[0]] = [k]  # mutual relation

        id_map_list = [-1] * self.configer.get('data', 'num_classes')[0]
        label_cnt = 0
        for i in range(self.configer.get('data', 'num_classes')[0]):
            if id_map_list[i] != -1:
                continue

            power = self.round / self.configer.get('merge', 'max_round')
            if self.configer.get('merge', 'enable_fscore') and \
                    fscore_list[i] / fscore_list[-1] < self.configer.get('merge', 'fscore_ratio') * power:
                continue

            id_map_list[i] = label_cnt
            if i in mutual_pairs_dict:
                for v in mutual_pairs_dict[i]:
                    assert id_map_list[v] == -1
                    id_map_list[v] = label_cnt

            label_cnt += 1

        fw = open('{}_{}'.format(self.configer.get('data', 'label_path'), self.round), 'w')
        with open(self.configer.get('data', 'label_path'), 'r') as fr:
            for line in fr.readlines():
                path, label = line.strip().split()
                if id_map_list[int(label)] == -1:
                    continue

                map_label = id_map_list[int(label)]
                fw.write('{} {}\n'.format(path, map_label))

        fw.close()
        shutil.move('{}_{}'.format(self.configer.get('data', 'label_path'), self.round),
                    self.configer.get('data', 'label_path'))
        shutil.copy(self.configer.get('data', 'label_path'),
                    os.path.join(self.configer.get('data', 'merge_dir'), 'label_{}.txt'.format(self.round)))
        old_label_cnt = self.configer.get('data', 'num_classes')[0]
        self.configer.update('data.num_classes', [label_cnt])
        return old_label_cnt - label_cnt

    def run(self):
        last_acc = 0.0
        while self.round <= self.configer.get('merge', 'max_round'):
            Log.info('Merge Round: {}'.format(self.round))
            Log.info('num classes: {}'.format(self.configer.get('data', 'num_classes')))
            self._init()
            self.train()
            acc, cmatrix, fscore_list = self.val(self.cls_data_loader.get_valloader())
            merge_cnt = self._merge_class(cmatrix, fscore_list)
            if merge_cnt < self.configer.get('merge', 'cnt_thre') \
                    or (acc - last_acc) < self.configer.get('merge', 'acc_thre'):
                break

            last_acc = acc
            self.round += 1

        shutil.copy(self.configer.get('data', 'label_path'),
                    os.path.join(self.configer.get('data', 'merge_dir'), 'merge_label.txt'))
        self._init()
        self.train()
        Log.info('num classes: {}'.format(self.configer.get('data', 'num_classes')))

    def train(self):
        """
          Train function of every epoch during train phase.
        """
        self.cls_net.train()
        start_time = time.time()
        while self.runner_state['iters'] < self.solver_dict['max_iters']:
            # Adjust the learning rate after every epoch.
            self.runner_state['epoch'] += 1
            for i, data_dict in enumerate(self.train_loader):
                Trainer.update(self, solver_dict=self.solver_dict)
                self.data_time.update(time.time() - start_time)
                # Change the data type.
                # Forward pass.
                out = self.cls_net(data_dict)
                # Compute the loss of the train batch & backward.

                loss_dict = self.loss(out)
                loss = loss_dict['loss']
                self.train_losses.update(loss.item(), data_dict['img'].size(0))
                self.optimizer.zero_grad()
                loss.backward()
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
                             'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                                 self.runner_state['epoch'], self.runner_state['iters'],
                                 self.solver_dict['display_iter'],
                                 RunnerHelper.get_lr(self.optimizer), batch_time=self.batch_time,
                                 data_time=self.data_time, loss=self.train_losses))

                    self.batch_time.reset()
                    self.data_time.reset()
                    self.train_losses.reset()

                if self.solver_dict['lr']['metric'] == 'iters' and self.runner_state['iters'] == self.solver_dict['max_iters']:
                    self.val()
                    break

                # Check to val the current model.
                if self.runner_state['iters'] % self.solver_dict['test_interval'] == 0:
                    self.val()

    def val(self, loader=None):
        """
          Validation function during the train phase.
        """
        self.cls_net.eval()
        start_time = time.time()

        loader = self.val_loader if loader is None else loader
        list_y_true, list_y_pred = [], []
        with torch.no_grad():
            for j, data_dict in enumerate(loader):
                out = self.cls_net(data_dict)
                loss_dict = self.loss(out)
                out_dict, label_dict, _ = RunnerHelper.gather(self, out)
                # Compute the loss of the val batch.
                self.cls_running_score.update(out_dict, label_dict)
                y_true = label_dict['out0'].view(-1).cpu().numpy().tolist()
                y_pred = out_dict['out0'].max(1)[1].view(-1).cpu().numpy().tolist()
                list_y_true.extend(y_true)
                list_y_pred.extend(y_pred)

                self.val_losses.update(loss_dict['loss'].mean().item(), data_dict['img'].size(0))

                # Update the vars of the val phase.
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            RunnerHelper.save_net(self, self.cls_net, performance=self.cls_running_score.top1_acc.avg['out0'])
            self.runner_state['performance'] = self.cls_running_score.top1_acc.avg['out0']
            # Print the log info & reset the states.
            Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))
            Log.info('Test Set: {} images'.format(len(list_y_true)))
            Log.info('TestLoss = {loss.avg:.8f}'.format(loss=self.val_losses))
            Log.info('Top1 ACC = {}'.format(self.cls_running_score.top1_acc.avg['out0']))
            # Log.info('Top5 ACC = {}'.format(self.cls_running_score.get_top5_acc()))
            acc= self.cls_running_score.top1_acc.avg['out0']
            cmatrix = confusion_matrix(list_y_true, list_y_pred)
            fscore_str = classification_report(list_y_true, list_y_pred, digits=5)
            fscore_list = [float(line.strip().split()[-2])
                           for line in fscore_str.split('\n')[2:] if len(line.strip().split()) > 0]
            self.batch_time.reset()
            self.val_losses.reset()
            self.cls_running_score.reset()
            self.cls_net.train()
            return acc, cmatrix, fscore_list


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
