import numpy as np
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
from lib.utils.tools.logger import Logger as Log
import math
from datetime import datetime
import random

def random_sample(label_indices_dict, min_cnt, max_cnt, epoch=None):
    sample_indices_dict = dict()
    for label in label_indices_dict.keys():
        target_indices = np.array(label_indices_dict[label])
        if len(target_indices) >= min_cnt:
            if max_cnt == -1 or len(target_indices) <= max_cnt:
                label_indices = target_indices
            else:
                if epoch is not None:
                    np.random.seed(epoch)
                label_indices = np.random.choice(target_indices, size=max_cnt, replace=False)
        else:
            if epoch is not None:
                np.random.seed(epoch)
            random_indices = np.random.choice(target_indices, size=min_cnt-len(target_indices), replace=True)
            label_indices = np.append(target_indices, random_indices)

        sample_indices_dict[label] = label_indices.tolist()

    return sample_indices_dict

def random_sample_multilabel(label_indices_dict, min_cnt, max_cnt, epoch=None):
    sample_indices_dict = dict()
    rm_keys = []
    for label in label_indices_dict.keys():
        sample_indices_dict[label] = dict(pos=None, neg=None)
        # positive examples
        target_indices = np.array(label_indices_dict[label]['pos'])
        if len(target_indices) == 0:
            rm_keys.append(label)
            continue
        elif len(target_indices) >= min_cnt:
            if max_cnt == -1 or len(target_indices) <= max_cnt:
                label_indices = target_indices
            else:
                if epoch is not None:
                    np.random.seed(epoch)
                label_indices = np.random.choice(target_indices, size=max_cnt, replace=False)
        else:
            if epoch is not None:
                np.random.seed(epoch)
            random_indices = np.random.choice(target_indices, size=min_cnt-len(target_indices), replace=True)
            label_indices = np.append(target_indices, random_indices)
        num_pos = len(label_indices)
        sample_indices_dict[label]['pos'] = label_indices.tolist()
        # negative examples
        target_indices = np.array(label_indices_dict[label]['neg'])
        if len(target_indices) == 0:
            rm_keys.append(label)
            continue
        elif len(target_indices) == num_pos:
            label_indices = target_indices
        elif len(target_indices) > num_pos:
            if epoch is not None:
                np.random.seed(epoch)
            label_indices = np.random.choice(target_indices, size=num_pos, replace=False)
        else:
            if epoch is not None:
                np.random.seed(epoch)
            random_indices = np.random.choice(target_indices, size=num_pos-len(target_indices), replace=True)
            label_indices = np.append(target_indices, random_indices)
        num_neg = len(label_indices)
        sample_indices_dict[label]['neg'] = label_indices.tolist()
        # negative examples === postive examples
        assert(num_neg == num_pos), "negative examples should be equal to postive for each global_label"    

    for label in rm_keys:
        sample_indices_dict.pop(label)   

    return sample_indices_dict

class OnlineTripletSampler(BatchSampler):
    """
      BatchSampler - from a multilabel dataset with global_label and sub_label, sub_label={0, 1, 2} where
                     0 for objects from same class, 1 for similar, and 2 for none
      Returns batches of size n_classes (=1 for global_label) * n_samples (=batch_size)
      Making sure that one epoch will used all the data
    """
    def __init__(self, label_list, batch_size=64, min_cnt=0, max_cnt=-1, is_distributed=False):
        assert(is_distributed is True), "OnlineTripletSampler only support distributed training"
        self.label_list = label_list
        self.label_indices_dict = dict()
        for i, mlabel in enumerate(self.label_list):
            if mlabel[0] not in self.label_indices_dict:
                self.label_indices_dict[mlabel[0]] = {'pos':[], 'neg':[]}
            if mlabel[1] == 0:
                self.label_indices_dict[mlabel[0]]['pos'].append(i)
            else:
                self.label_indices_dict[mlabel[0]]['neg'].append(i)
       
        self.is_distributed = is_distributed
        self.n_classes = 1
        self.n_samples = batch_size // self.n_classes
        self.min_cnt = min_cnt
        self.max_cnt = max_cnt
        assert batch_size % self.n_samples == 0
        sample_dict = random_sample_multilabel(self.label_indices_dict, self.min_cnt, self.max_cnt)
        self.num_samples = 0
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        for k in sample_dict:
            self.num_samples += len(sample_dict[k]['pos']) + len(sample_dict[k]['neg'])

        Log.info('OnlineTripletSampler: The number of resampled images pergpu is {} by min_cnt={} & max_cnt={}...'.format(self.num_samples, self.min_cnt, self.max_cnt))

    def __iter__(self):
        samples_indices_dict = random_sample_multilabel(self.label_indices_dict, self.min_cnt, self.max_cnt)
        valid_dict = {label: True for label in samples_indices_dict.keys()}
        label_samples_dict = {label: 0 for label in samples_indices_dict.keys()}
        for l in samples_indices_dict.keys():
            np.random.shuffle(samples_indices_dict[l]['pos'])
            np.random.shuffle(samples_indices_dict[l]['neg'])

        n_samples = self.n_samples // 2
        while len(valid_dict.keys()) >= self.n_classes:
            classes = np.random.choice(list(valid_dict.keys()), self.n_classes, replace=False)
            indices = []
            for c in classes:
                indices.extend(samples_indices_dict[c]['pos'][label_samples_dict[c]:label_samples_dict[c]+n_samples])
                indices.extend(samples_indices_dict[c]['neg'][label_samples_dict[c]:label_samples_dict[c]+n_samples])
                label_samples_dict[c] += n_samples
                if label_samples_dict[c] + n_samples > len(samples_indices_dict[c]['pos']):
                    del valid_dict[c]

            yield indices

    def __len__(self):
        return self.num_samples // (self.n_samples * self.n_classes)


class RankingSampler(BatchSampler):
    """
      BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
      Returns batches of size n_classes * n_samples
      Making sure that one epoch will used all the data
    """
    def __init__(self, label_list, samples_per_class=2, batch_size=64, min_cnt=0, max_cnt=-1, is_distributed=False):
        self.label_list = label_list
        self.label_indices_dict = dict()
        for i, mlabel in enumerate(self.label_list):
            if mlabel[0] not in self.label_indices_dict:
                self.label_indices_dict[mlabel[0]] = [i]
            else:
                self.label_indices_dict[mlabel[0]].append(i)
       
        self.is_distributed = is_distributed
        self.n_classes = batch_size // samples_per_class
        self.n_samples = samples_per_class
        self.min_cnt = min_cnt
        self.max_cnt = max_cnt
        assert batch_size % samples_per_class == 0
        sample_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        self.num_samples = 0
        self.epoch = 0
        if self.is_distributed:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            for k in sample_dict:
                num_samples_pergpu = int(math.ceil(len(sample_dict[k]) * 1.0 / self.num_replicas)) 
                self.num_samples += num_samples_pergpu
        else:
            for k in sample_dict:
                self.num_samples += len(sample_dict[k])

        Log.info('RankingSampler: The number of resampled images pergpu is {} by min_cnt={} & max_cnt={}...'.format(self.num_samples, self.min_cnt, self.max_cnt))

    def __iter__(self):
        valid_dict = {label: True for label in self.label_indices_dict.keys()}
        label_samples_dict = {label: 0 for label in self.label_indices_dict.keys()}
        if self.is_distributed:
            self.set_epoch()
            samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt, epoch=self.epoch)
            for k in samples_indices_dict:
                num_samples_pergpu = int(math.ceil(len(samples_indices_dict[k]) * 1.0 / self.num_replicas))
                np.random.seed(self.epoch)
                samples_indices_dict[k] = samples_indices_dict[k] + np.random.choice(samples_indices_dict[k], num_samples_pergpu*self.num_replicas-len(samples_indices_dict[k]), replace=True).tolist()
                np.random.seed(self.epoch)
                np.random.shuffle(samples_indices_dict[k])
                offset = num_samples_pergpu * self.rank
                samples_indices_dict[k] = samples_indices_dict[k][offset : offset + num_samples_pergpu]
        else:
            samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        for l in samples_indices_dict.keys():
            np.random.shuffle(samples_indices_dict[l])

        while len(valid_dict.keys()) >= self.n_classes:
            classes = np.random.choice(list(valid_dict.keys()), self.n_classes, replace=False)
            indices = []
            for c in classes:
                indices.extend(samples_indices_dict[c][label_samples_dict[c]:label_samples_dict[c]+self.n_samples])
                label_samples_dict[c] += self.n_samples
                if label_samples_dict[c] + self.n_samples > len(samples_indices_dict[c]):
                    del valid_dict[c]

            yield indices

    def __len__(self):
        return self.num_samples // (self.n_samples * self.n_classes)

    def set_epoch(self):
        self.epoch += 1

class BalanceSampler(BatchSampler):
    def __init__(self, label_list, batch_size=64, min_cnt=0, max_cnt=-1, is_distributed=False):
        self.label_list = label_list
        self.label_indices_dict = dict()
        for i, mlabel in enumerate(self.label_list):
            if mlabel[0] not in self.label_indices_dict:
                self.label_indices_dict[mlabel[0]] = [i]
            else:
                self.label_indices_dict[mlabel[0]].append(i)

        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.min_cnt = min_cnt
        self.max_cnt = max_cnt
        sample_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        self.num_samples = 0
        self.epoch = 0
        if self.is_distributed:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            for k in sample_dict:
                num_samples_pergpu = int(math.ceil(len(sample_dict[k]) * 1.0 / self.num_replicas)) 
                self.num_samples += num_samples_pergpu
        else:
            for k in sample_dict:
                self.num_samples += len(sample_dict[k])

        Log.info('BalanceSampler: The number of resampled images pergpu is {} by min_cnt={} & max_cnt={}...'.format(self.num_samples, self.min_cnt, self.max_cnt))

    def __iter__(self):
        if self.is_distributed:
            self.set_epoch()
            samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt, epoch=self.epoch)
            for k in samples_indices_dict:
                num_samples_pergpu = int(math.ceil(len(samples_indices_dict[k]) * 1.0 / self.num_replicas))
                np.random.seed(self.epoch)
                samples_indices_dict[k] = samples_indices_dict[k] + np.random.choice(samples_indices_dict[k], num_samples_pergpu*self.num_replicas-len(samples_indices_dict[k]), replace=True).tolist()
                np.random.seed(self.epoch)
                np.random.shuffle(samples_indices_dict[k])
                offset = num_samples_pergpu * self.rank
                samples_indices_dict[k] = samples_indices_dict[k][offset : offset + num_samples_pergpu]
        else:
            samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        samples_indices = []
        for k in samples_indices_dict:
            samples_indices.extend(samples_indices_dict[k])

        random.seed(datetime.now())
        random.shuffle(samples_indices)
        sample_index = 0
        assert len(samples_indices) > self.batch_size
        while sample_index + self.batch_size < len(samples_indices):
            yield samples_indices[sample_index:sample_index+self.batch_size]
            sample_index += self.batch_size

    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self):
        self.epoch += 1

class ReverseSampler(BatchSampler):
    def __init__(self, label_list, batch_size=64, min_cnt=0, max_cnt=-1, is_distributed=False):
        self.label_list = label_list
        self.label_indices_dict = dict()
        for i, mlabel in enumerate(self.label_list):
            if mlabel[0] not in self.label_indices_dict:
                self.label_indices_dict[mlabel[0]] = [i]
            else:
                self.label_indices_dict[mlabel[0]].append(i)
        real_min_cnt = 100000
        for k in self.label_indices_dict:
            if real_min_cnt > len(self.label_indices_dict[k]):
                real_min_cnt = len(self.label_indices_dict[k])

        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.min_cnt = max(min_cnt, real_min_cnt)
        self.max_cnt = max(max_cnt, real_min_cnt, min_cnt)
        sample_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        self.num_samples = 0
        self.epoch = 0
        if self.is_distributed:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            for k in sample_dict:
                num_samples_pergpu = int(math.ceil(len(sample_dict[k]) * 1.0 / self.num_replicas)) 
                self.num_samples += num_samples_pergpu
        else:
            for k in sample_dict:
                self.num_samples += len(sample_dict[k])

        Log.info('ReverseSampler: The number of resampled images pergpu is {} by min_cnt={} & max_cnt={}...'.format(self.num_samples, self.min_cnt, self.max_cnt))

    def __iter__(self):
        if self.is_distributed:
            self.set_epoch()
            samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt, epoch=self.epoch)
            for k in samples_indices_dict:
                num_samples_pergpu = int(math.ceil(len(samples_indices_dict[k]) * 1.0 / self.num_replicas))
                np.random.seed(self.epoch)
                samples_indices_dict[k] = samples_indices_dict[k] + np.random.choice(samples_indices_dict[k], num_samples_pergpu*self.num_replicas-len(samples_indices_dict[k]), replace=True).tolist()
                np.random.seed(self.epoch)
                np.random.shuffle(samples_indices_dict[k])
                offset = num_samples_pergpu * self.rank
                samples_indices_dict[k] = samples_indices_dict[k][offset : offset + num_samples_pergpu]
        else:
            samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        samples_indices = []
        for k in samples_indices_dict:
            samples_indices.extend(samples_indices_dict[k])

        np.random.shuffle(samples_indices)
        sample_index = 0
        assert len(samples_indices) > self.batch_size
        while sample_index + self.batch_size < len(samples_indices):
            yield samples_indices[sample_index:sample_index+self.batch_size]
            sample_index += self.batch_size

    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self):
        self.epoch += 1

class ReverseSampler_BKUP(BatchSampler):
    def __init__(self, label_list, batch_size=64, min_cnt=0, max_cnt=-1, is_distributed=False):
        self.label_list = label_list
        self.label_indices_dict = dict()
        for i, mlabel in enumerate(self.label_list):
            if mlabel[0] not in self.label_indices_dict:
                self.label_indices_dict[mlabel[0]] = [i]
            else:
                self.label_indices_dict[mlabel[0]].append(i)

        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.min_cnt = min_cnt
        self.max_cnt = max_cnt
        sample_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        self.num_samples = 0
        self.epoch = 0
        self.class_rweight = []
        if self.is_distributed:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            for k in sample_dict:
                num_samples_pergpu = int(math.ceil(len(sample_dict[k]) * 1.0 / self.num_replicas)) 
                self.num_samples += num_samples_pergpu
                self.class_rweight.append(num_samples_pergpu + 0.0)
        else:
            for k in sample_dict:
                self.num_samples += len(sample_dict[k])
                self.class_rweight.append(len(sample_dict[k]) + 0.0)

        max_weight = max(self.class_rweight)
        sum_weight = 0.0
        for i in range(len(self.class_rweight)):
            self.class_rweight[i] = max_weight / self.class_rweight[i]
            sum_weight += self.class_rweight[i]
        for i in range(len(self.class_rweight)):
            self.class_rweight[i] /= sum_weight 

        Log.info('ReverseSampler: The number of resampled images is {}...'.format(self.num_samples))

    def __iter__(self):
        if self.is_distributed:
            self.set_epoch()
            samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt, epoch=self.epoch)
            for k in samples_indices_dict:
                num_samples_pergpu = int(math.ceil(len(samples_indices_dict[k]) * 1.0 / self.num_replicas))
                np.random.seed(self.epoch)
                samples_indices_dict[k] = samples_indices_dict[k] + np.random.choice(samples_indices_dict[k], num_samples_pergpu*self.num_replicas-len(samples_indices_dict[k]), replace=True).tolist()
                np.random.seed(self.epoch)
                np.random.shuffle(samples_indices_dict[k])
                offset = num_samples_pergpu * self.rank
                samples_indices_dict[k] = samples_indices_dict[k][offset : offset + num_samples_pergpu]
        else:
            samples_indices_dict = random_sample(self.label_indices_dict, self.min_cnt, self.max_cnt)
        samples_indices = []
        sample_classes = np.random.choice(list(samples_indices_dict.keys()), self.num_samples, replace=True, p=self.class_rweight)
        for k in samples_indices_dict:
            samples_indices.extend(np.random.choice(samples_indices_dict[k], (sample_classes == k).sum(), replace=True).tolist())
        assert(len(samples_indices) == self.num_samples)
        np.random.shuffle(samples_indices)
        np.save('rank{}_epoch{}.npy'.format(self.rank, self.epoch), samples_indices)
        sample_index = 0
        assert len(samples_indices) > self.batch_size
        while sample_index + self.batch_size < len(samples_indices):
            yield samples_indices[sample_index:sample_index+self.batch_size]
            sample_index += self.batch_size

    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self):
        self.epoch += 1
