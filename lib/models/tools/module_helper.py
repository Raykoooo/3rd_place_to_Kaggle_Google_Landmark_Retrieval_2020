#!/usr/bin/env python
# -*- coding:utf-8 -*-


import functools
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.tools.metric_linear import (ArcLinear, CosineLinear, Linear,
                                            SphereLinear)
from lib.utils.tools.logger import Logger as Log
from lib.utils.aggregator import (crow, gap, gem, gmp, rmac, rmac_simple, scda, spoc)

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


class ModuleHelper(object):

    @staticmethod
    def concat(data_dict, key_name):
        out = []
        for key in data_dict:
            if key_name in key:
                out.append(data_dict[key])
        assert len(out) > 0
        return torch.cat(out, 0)

    @staticmethod
    def preprocess(x, normalize):
        style = normalize['norm_style']

        if style in normalize['norm_dict']:
            norm_dict = normalize['norm_dict'][style]
        else:
            raise Exception('Not implemented Error!!!')

        x = x.div(norm_dict['div_value'])
        x = x - torch.cuda.FloatTensor(norm_dict['mean']).view(1, 3, 1, 1)
        x = x.div(torch.cuda.FloatTensor(norm_dict['std']).view(1, 3, 1, 1))
        return x
    @staticmethod
    def postprocess(feat, method='GAP', bn=None):
        if method == 'CROW':
            feat = crow(feat)
        elif method == 'GAP':
            feat = gap(feat)
        elif method == 'GEM':
            feat = gem(feat)
        elif method == 'GMP':
            feat = gmp(feat)
        elif method == 'RMAC':
            feat = rmac(feat)
        elif method == 'RMAC_SIMPLE':
            feat = rmac_simple(feat)
        elif method == 'SCDA':
            feat = scda(feat)
        elif method == 'SPOC':
            feat = spoc(feat)
        elif method == 'NONE':
            pass
        else:
            raise Exception('Not implemented Error!!!')

        feat = bn(feat) if bn else feat
        return feat

    @staticmethod
    def get_score(logits, method='softmax'):
        if method == 'softmax':
            score = F.softmax(logits, dim=1)

        elif method == 'sigmoid':
            score = F.sigmoid(logits)

        else:
            raise Exception('Not implemented Error!!!')

        return score

    @staticmethod
    def normalize(feat, method='L2'):
        if method == 'L1':
            feat = feat / torch.sum(torch.abs(feat), dim=1, keepdim=True)
        elif method == 'L2':
            feat = feat / torch.sqrt(torch.sum(feat**2, dim=1, keepdim=True))
        elif method == 'POWER':
            ppp = 0.3
            feat = torch.sign(feat) * (torch.abs(feat) ** ppp)
        elif method == 'NONE':
            return feat
        else:
            Log.error('Norm Type {} is invalid.'.format(type))
            exit(1)

        return feat

    @staticmethod
    def Linear(linear_type):
        if linear_type == 'default':
            return Linear

        if linear_type == 'nobias':
            return functools.partial(Linear, bias=False)

        elif 'arc' in linear_type:
            #example arc0.5_64  arc0.32_64 easyarc0.5_64
            margin_scale = linear_type.split('arc')[1]
            margin = float(margin_scale.split('_')[0])
            scale = float(margin_scale.split('_')[1])
            easy = True if 'easy' in linear_type else False
            return functools.partial(ArcLinear, s=scale, m=margin, easy_margin=easy)

        elif linear_type == 'cos0.4_30':
            return functools.partial(CosineLinear, s=30, m=0.5)

        elif linear_type == 'cos0.4_64':
            return functools.partial(CosineLinear, s=64, m=0.5)

        elif linear_type == 'sphere4':
            return functools.partial(SphereLinear, m=4)

        else:
            Log.error('Not support linear type: {}.'.format(linear_type))
            exit(1)

    @staticmethod
    def BNReLU(num_features, norm_type=None, **kwargs):
        if norm_type == 'batchnorm':
            return nn.Sequential(
                nn.BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif norm_type == 'sync_batchnorm':
            from lib.extensions.ops.sync_bn import BatchNorm2d
            return nn.Sequential(
                BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif norm_type == 'instancenorm':
            return nn.Sequential(
                nn.InstanceNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        else:
            Log.error('Not support BN type: {}.'.format(norm_type))
            exit(1)

    @staticmethod
    def BatchNorm2d(norm_type=None):
        if norm_type == 'batchnorm':
            return nn.BatchNorm2d

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm2d
        # elif bn_type == 'inplace_abn':
        #    from extensions.ops.inplace_abn.bn import InPlaceABNSync
        #    if ret_cls:
        #        return InPlaceABNSync

        #    return functools.partial(InPlaceABNSync, activation='none')

        else:
            Log.error('Not support BN type: {}.'.format(norm_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True):
        if pretrained is None:
            return model

        if not os.path.exists(pretrained):
            Log.info('{} not exists.'.format(pretrained))
            return model

        if all_match:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location="cpu")
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'prefix.{}'.format(k) in model_dict:
                    load_dict['prefix.{}'.format(k)] = v
                else:
                    load_dict[k] = v

            # load_dict = {k: v for k, v in pretrained_dict.items() if 'resinit.{}'.format(k) not in model_dict}
            model.load_state_dict(load_dict)

        else:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = model.state_dict()
            load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            Log.info('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_tf_efficientnet_model(model, pretrained=None, strict=False):
        if pretrained is None:
            return model

        if len(glob.glob(pretrained + '*')) == 0 or not os.path.exists(glob.glob(pretrained + '*')[0]):
            Log.info('{} not exists.'.format(pretrained))
            return model

        import pdb
        from tensorflow.python import pywrap_tensorflow
        Log.info('Loading pretrained model:{}'.format(pretrained))
        tf_reader = pywrap_tensorflow.NewCheckpointReader(pretrained)
        for tf_key in tf_reader.get_variable_to_shape_map():
            if tf_key.startswith('efficientnet'):
                model_name = tf_key.split('/')[0]
                break
        model_dict = model.state_dict()
        load_dict = dict()
        bn_list = [['running_mean', 'moving_mean'], ['running_var', 'moving_variance'], ['weight', 'gamma'],
                   ['bias', 'beta']]
        # stem and head block
        for block_name in ['stem', 'head']:
            key = '_conv_{}.weight'.format(block_name)
            tf_key = '{}/{}/conv2d/kernel'.format(model_name, block_name)
            assert (key in model_dict)
            tf_value = tf_reader.get_tensor(tf_key)
            load_dict[key] = torch.from_numpy(tf_value.transpose(3, 2, 0, 1))
            for bn_name in bn_list:
                key = '_bn_{}.{}'.format(block_name, bn_name[0])
                tf_key = '{}/{}/tpu_batch_normalization/{}'.format(model_name, block_name, bn_name[1])
                assert (key in model_dict)
                tf_value = tf_reader.get_tensor(tf_key)
                load_dict[key] = torch.from_numpy(tf_value)
        # MBConvBlocks
        module_list = [[['_expand_conv', '_project_conv'], 'conv2d'], [['_depthwise_conv'], 'depthwise_conv2d'],
                       [['_se_reduce', '_se_expand'], 'se/conv2d'],
                       [['_expand_bn', '_depthwise_bn', '_project_bn'], 'tpu_batch_normalization']]
        mb_block_id = 0
        mb_block_flag = True
        while mb_block_flag:
            for module in module_list:
                key_id = 0
                tf_key_part = module[1]
                for key_part in module[0]:
                    if key_part.startswith('_depthwise_conv'):
                        key = '_blocks.{}.{}.weight'.format(mb_block_id, key_part)
                        tf_key = '{}/blocks_{}/{}/depthwise_kernel'.format(model_name, mb_block_id, tf_key_part)
                        if key not in model_dict:
                            mb_block_flag = False
                            Log.info('Ignore parameter: {} <---> {}'.format(key, tf_key))
                            continue
                        tf_value = tf_reader.get_tensor(tf_key)
                        load_dict[key] = torch.from_numpy(tf_value.transpose(2, 3, 0, 1))
                    elif key_part.endswith('_conv'):
                        key = '_blocks.{}.{}.weight'.format(mb_block_id, key_part)
                        if key_id > 0:
                            tf_key = '{}_{}'.format(tf_key_part, key_id)
                        else:
                            tf_key = '{}'.format(tf_key_part)
                        tf_key = '{}/blocks_{}/{}/kernel'.format(model_name, mb_block_id, tf_key)
                        if key not in model_dict:
                            Log.info('Ignore parameter: {} <---> {}'.format(key, tf_key))
                            continue
                        tf_value = tf_reader.get_tensor(tf_key)
                        load_dict[key] = torch.from_numpy(tf_value.transpose(3, 2, 0, 1))
                        key_id += 1
                    elif key_part.startswith('_se'):
                        key = '_blocks.{}.{}.weight'.format(mb_block_id, key_part)
                        if key_id > 0:
                            tf_key = '{}_{}'.format(tf_key_part, key_id)
                        else:
                            tf_key = '{}'.format(tf_key_part)
                        tf_key = '{}/blocks_{}/{}/kernel'.format(model_name, mb_block_id, tf_key)
                        if key not in model_dict:
                            Log.info('Ignore parameter: {} <---> {}'.format(key, tf_key))
                            continue
                        tf_value = tf_reader.get_tensor(tf_key)
                        load_dict[key] = torch.from_numpy(tf_value.transpose(3, 2, 0, 1))

                        key = '_blocks.{}.{}.bias'.format(mb_block_id, key_part)
                        if key_id > 0:
                            tf_key = '{}_{}'.format(tf_key_part, key_id)
                        else:
                            tf_key = '{}'.format(tf_key_part)
                        tf_key = '{}/blocks_{}/{}/bias'.format(model_name, mb_block_id, tf_key)
                        tf_value = tf_reader.get_tensor(tf_key)
                        load_dict[key] = torch.from_numpy(tf_value)
                        key_id += 1
                    elif key_part.endswith('_bn'):
                        bn_flag = False
                        for bn_name in bn_list:
                            key = '_blocks.{}.{}.{}'.format(mb_block_id, key_part, bn_name[0])
                            if key_id > 0:
                                tf_key = '{}_{}'.format(tf_key_part, key_id)
                            else:
                                tf_key = '{}'.format(tf_key_part)
                            tf_key = '{}/blocks_{}/{}/{}'.format(model_name, mb_block_id, tf_key, bn_name[1])
                            if key not in model_dict:
                                Log.info('Ignore parameter: {} <---> {}'.format(key, tf_key))
                                continue
                            bn_flag = True
                            tf_value = tf_reader.get_tensor(tf_key)
                            load_dict[key] = torch.from_numpy(tf_value)
                        if bn_flag:
                            key_id += 1
            mb_block_id += 1
        unexpected_keys = []
        unmatched_keys = []
        own_state = model.state_dict()
        for name, param in load_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                if strict:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(),
                                               param.size()))
                else:
                    unmatched_keys.append(name)
        missing_keys = set(own_state.keys()) - set(load_dict.keys())

        err_msg = []
        if unexpected_keys:
            err_msg.append('unexpected key in source state_dict: {}'.format(', '.join(unexpected_keys)))
        if missing_keys:
            err_msg.append('missing keys in source state_dict: {}'.format(', '.join(missing_keys)))
        if unmatched_keys:
            err_msg.append('unmatched keys in source state_dict: {}'.format(', '.join(unmatched_keys)))
        err_msg = '\n'.join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)
            else:
                Log.warn(err_msg)
        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.PyTorchCV', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        Log.info('Loading pretrained model:{}'.format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def cam(conv_out, score, fc, reduction):
        # fc weight and bias
        _, idx = torch.max(score,1)
        fc_weight = fc._parameters["weight"][idx]
        _c, c = fc_weight.size()
        fc_weight = fc_weight.reshape((_c, c, 1, 1))
        
        if reduction != None:
            # reduce weight and bias
            reduction_weight = reduction._parameters['weight'].transpose(1,0)
            reduction_bias = reduction._parameters['bias']
            fc_weight -= reduction_bias.reshape(fc_weight.size())
            cam_weight = F.conv2d(fc_weight, weight=reduction_weight)
        else:
            cam_weight = fc_weight

        # cam 
        cam = F.relu(F.conv2d(conv_out, weight=cam_weight))
        
        # normalization
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 0.00001)

        return cam
