#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Main Scripts for computer vision.


import argparse
import functools
import json
import os
import random
import sys

import torch
import torch.backends.cudnn as cudnn
from lib.utils.tools.configer import Configer
from lib.utils.tools.logger import Logger as Log


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str,
                        dest='config_file', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default="train", type=str,
                        dest='phase', help = 'The file of the hyper parameters.')
    parser.add_argument('--dtype', default="none", type=str,
                        dest='dtype', help='The dtype of the network.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')
    parser.add_argument('--workers', default=None, type=int,
                        dest='data.workers', help='The number of workers to load data.')
    parser.add_argument('--num_data_sources', default=1, type=int,
                        dest='data.num_data_sources', help='The number of data sources for training.')
    parser.add_argument('--norm_style', default="pytorch", type=str,
                        dest='data.normalize.norm_style', help='The norm_style of preprocess.')
    num_data_sources = int(sys.argv[sys.argv.index('--num_data_sources') + 1])
    for source in range(num_data_sources):
        parser.add_argument('--src{}_data_dir'.format(source), default="/", type=str,
                            dest='data.src{}_data_dir'.format(source), help='The Directory of the data.')
        parser.add_argument('--src{}_val_data_dir'.format(source), default=None, type=str,
                            dest='data.src{}_val_data_dir'.format(source), help='The Directory of the val data.')
        parser.add_argument('--src{}_num_classes'.format(source), default=None, nargs='+', type=int,
                            dest='data.src{}_num_classes'.format(source), help='The number of classes.')
        parser.add_argument('--src{}_relabel'.format(source), type=str2bool, nargs='?', default=False,
                            dest='data.src{}_relabel'.format(source), help='Whether to relabel the dataset.')
        parser.add_argument('--src{}_label_path'.format(source), default="", type=str,
                            dest='data.src{}_label_path'.format(source), help='The Label path of the data.')
        parser.add_argument('--src{}_val_label_path'.format(source), default=None, type=str,
                            dest='data.src{}_val_label_path'.format(source), help='The Label path of the val data.')
        parser.add_argument('--src{}_train_batch_size'.format(source), default=None, type=int,
                            dest='train.src{}_batch_size'.format(source), help='The batch size of training.')
        parser.add_argument('--src{}_val_batch_size'.format(source), default=None, type=int,
                            dest='val.src{}_batch_size'.format(source), help='The batch size of validation.')
        parser.add_argument('--src{}_train_loader'.format(source), default=None, type=str,
                            dest='train.src{}_loader'.format(source), help='The train loader type.')
        parser.add_argument('--src{}_val_loader'.format(source), default=None, type=str,
                            dest='val.src{}_loader'.format(source), help='The aux loader type.')
        parser.add_argument('--src{}_samples_per_class'.format(source), default=None, type=int,
                            dest='train.src{}_samples_per_class'.format(source), help='The number of samples per-class.')
        parser.add_argument('--src{}_min_count'.format(source), default=0, type=int,
                            dest='train.src{}_min_count'.format(source), help='The min count of per-sku.')
        parser.add_argument('--src{}_max_count'.format(source), default=-1, type=int,
                            dest='train.src{}_max_count'.format(source), help='The max count of per-sku.')
    parser.add_argument('--val_ratio', default=0.1, type=float,
                        dest='data.val_ratio', help='The val ratio for validation.')
    parser.add_argument('--include_val', type=str2bool, nargs='?', default=False,
                        dest='data.include_val', help='Include validation set for final training.')
    parser.add_argument('--global_num_classes', default=None, type=int,
                        dest='data.global_num_classes', help='The number of global classes.')
    parser.add_argument('--mixup', type=str2bool, nargs='?', default=False,
                        dest='data.mixup', help='mixup training. src0 and src1 load the same dataset.')
    parser.add_argument('--multiscale', default=None, nargs='+', type=float,
                        dest='data.multiscale', help='multiscale training.')
    # ***********  Params for augmentations.  **********
    parser.add_argument('--shuffle_trans_seq', default=None, nargs='+', type=str,
                        dest='train.aug_trans.shuffle_trans_seq', help='The augmentations transformation sequence.')
    parser.add_argument('--trans_seq', default=None, nargs='+', type=str,
                        dest='train.aug_trans.trans_seq', help='The augmentations transformation sequence.')

    for stream in ['', 'main_', 'peer_']:
        # ***********  Params for distilling.  **********
        parser.add_argument('--{}backbone'.format(stream), default=None, type=str,
                            dest='network.{}backbone'.format(stream), help='The main base network of model.')
        parser.add_argument('--{}rm_last_stride'.format(stream), type=str2bool, nargs='?', default=False,
                            dest='network.{}rm_last_stride'.format(stream), help='Whether to set last_stride=1 instead of 2.')
        parser.add_argument('--{}rm_last_bottleneck'.format(stream), type=str2bool, nargs='?', default=False,
                            dest='network.{}rm_last_bottleneck'.format(stream), help='Whether to remove last bottleck of layer4')
        parser.add_argument('--{}pretrained'.format(stream), type=str, default=None,
                            dest='network.{}pretrained'.format(stream), help='The path to peer pretrained model.')
        for branch in ['', 'bfe_']:
            parser.add_argument('--{}{}fc_dim'.format(stream, branch), default=None, type=int,
                                dest='network.{}{}fc_dim'.format(stream, branch), help='The dim of reduction features.')
            parser.add_argument('--{}{}fc_bn'.format(stream, branch), type=str2bool, nargs='?', default=True,
                                dest='network.{}{}fc_bn'.format(stream, branch), help='Whether to bn fc features.')
            parser.add_argument('--{}{}fc_relu'.format(stream, branch), type=str2bool, nargs='?', default=True,
                                dest='network.{}{}fc_relu'.format(stream, branch), help='Whether to relu fc features.')
            parser.add_argument('--{}{}feat_dim'.format(stream, branch), default=256, type=int,
                                dest='network.{}{}feat_dim'.format(stream, branch), help='The dim of embedding features.')
            parser.add_argument('--{}{}embed'.format(stream, branch), type=str2bool, nargs='?', default=True,
                                dest='network.{}{}embed'.format(stream, branch), help='Whether to embed features.')
            parser.add_argument('--{}{}embed_norm_type'.format(stream, branch), type=str, default=None,
                                dest='network.{}{}embed_norm_type'.format(stream, branch), help='normalize embed features from None, BN, L2')
            parser.add_argument('--{}{}global_linear_type'.format(stream, branch), default='default', type=str,
                                dest='network.{}{}global_linear_type'.format(stream, branch), help='The linear type of the network.')
            for source in range(num_data_sources):
                parser.add_argument('--{}{}src{}_linear_type'.format(stream, branch, source), default='default', type=str,
                                    dest='network.{}{}src{}_linear_type'.format(stream, branch, source), help='The linear type of the network.')

    # ***********  Params for model.  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network.model_name', help='The name of model.')
    parser.add_argument('--checkpoints_dir', default=None, type=str,
                        dest='network.checkpoints_dir', help='The root dir of model save path.')
    parser.add_argument('--checkpoints_name', default=None, type=str,
                        dest='network.checkpoints_name', help='The name of checkpoint model.')
    parser.add_argument('--norm_type', default=None, type=str,
                        dest='network.norm_type', help='The BN type of the network.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network.resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network.resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network.resume_continue', help='Whether to continue training.')
    parser.add_argument('--embed_after_norm', type=str2bool, nargs='?', default=True,
                        dest='network.embed_after_norm', help='place embed layer after normalization.')
    parser.add_argument('--resume_val', type=str2bool, nargs='?', default=False,
                        dest='network.resume_val', help='Whether to validate during resume.')
    parser.add_argument('--gather', type=str2bool, nargs='?', default=True,
                        dest='network.gather', help='Whether to gather the output of model.')
    parser.add_argument('--syncbn', type=str2bool, nargs='?', default=False,
                        dest='network.syncbn', help='Whether to use syncbn.')
    parser.add_argument('--bb_lr_scale', default=1.0, type=float,
                        dest='network.bb_lr_scale', help='The backbone LR scale.')
    parser.add_argument('--clip_grad', type=str2bool, nargs='?', default=False,
                        dest='network.clip_grad', help='Whether to clip grad?')
    parser.add_argument('--distill_method', default=None, type=str,
                        dest='network.distill_method', help='The distill method.')

    # ***********  Params for solver.  **********
    parser.add_argument('--solver', default="solver", type=str,
                        dest='train.solver', help='The train loader type.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='solver.lr.base_lr', help='The learning rate.')
    parser.add_argument('--is_warm', type=str2bool, nargs='?', default=False,
                        dest='solver.lr.is_warm', help='Whether to warm-up for training.')
    parser.add_argument('--warm_iters', default=None, type=int,
                        dest='solver.lr.warm.warm_iters', help='The warm-up iters of training.')
    parser.add_argument('--warm_freeze', type=str2bool, nargs='?', default=False,
                        dest='solver.lr.warm.freeze', help='Whether to freeze backbone when is_warm=True')
    parser.add_argument('--max_iters', default=None, type=int,
                        dest='solver.max_iters', help='The max iters of training.')
    parser.add_argument('--display_iter', default=None, type=int,
                        dest='solver.display_iter', help='The display iteration of train logs.')
    parser.add_argument('--test_interval', default=None, type=int,
                        dest='solver.test_interval', help='The test interval of validation.')
    parser.add_argument('--save_iters', default=None, type=int,
                        dest='solver.save_iters', help='The saving iters of checkpoint model.')

    # ***********  Params for Optim Method.  **********
    parser.add_argument('--optim_method', default=None, type=str,
                        dest='solver.optim.optim_method', help='The optim method that used.')
    parser.add_argument('--sgd_wd', default=None, type=float,
                        dest='solver.optim.sgd.weight_decay', help='The weight decay for SGD.')
    parser.add_argument('--nesterov', type=str2bool, nargs='?', default=False,
                        dest='solver.optim.sgd.nesterov', help='The weight decay for SGD.')
    parser.add_argument('--adam_wd', default=None, type=float,
                        dest='solver.optim.adam.weight_decay', help='The weight decay for Adam.')

    # ***********  Params for LR Policy.  **********
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='solver.lr.lr_policy', help='The policy of lr during training.')
    parser.add_argument('--step_value', default=None, nargs='+', type=int,
                        dest='solver.lr.multistep.step_value', help='The step values for multistep.')
    parser.add_argument('--gamma', default=None, type=float,
                        dest='solver.lr.multistep.gamma', help='The gamma for multistep.')
    parser.add_argument('--power', default=None, type=float,
                        dest='solver.lr.lambda_poly.power', help='The power for lambda poly.')
    parser.add_argument('--max_power', default=None, type=float,
                        dest='solver.lr.lambda_range.max_power', help='The power for lambda range.')

    # ***********  Params for loss.  **********
    parser.add_argument('--loss_type', default=None, type=str,
                        dest='loss.loss_type', help='The loss type of the network.')

    # ***********  Params for gan:bfe.  **********
    parser.add_argument('--gan_method', default="batch_drop", type=str,
                        dest='bfe.gan_method', help='select gan_method for bfe')
    parser.add_argument('--height_ratio', default=0.3, type=float,
                        dest='bfe.height_ratio', help='crop height ratio for batch feature erasing')
    parser.add_argument('--width_ratio', default=0.3, type=float,
                        dest='bfe.width_ratio', help='crop width ratio for batch feature erasing')
    parser.add_argument('--drop_prob', default=0.25, type=float,
                        dest='bfe.drop_prob', help='drop probability for dropblock method')
    parser.add_argument('--drop_block_size', default=5, type=float,
                        dest='bfe.drop_block_size', help='drop block_size for dropblock method')
    parser.add_argument('--drop_prob_start', default=0, type=float,
                        dest='bfe.drop_prob_start', help='starting prob for LinearScheduler method')
    parser.add_argument('--drop_prob_nsteps', default=1000, type=int,
                        dest='bfe.drop_prob_nsteps', help='nsteps from start prob to prob')
    parser.add_argument('--bfe_concat', type=str2bool, nargs='?', default=False,
                        dest='bfe.concat', help='Whether to use concat.')

    # ***********  Params for bbn  **********
    parser.add_argument('--num_groups', default=200, type=int,
                        dest='bbn.num_groups', help='divide max_iters to n groups')
    parser.add_argument('--mixup_init_iter', default=0, type=int,
                        dest='bbn.mixup_init_iter', help='initialize self.iter for alpha')
    parser.add_argument('--mixup_start_iter', default=0, type=int,
                        dest='bbn.mixup_start_iter', help='when to start mixup')

    # ***********  Params for merging.  **********
    parser.add_argument('--merge_dir', default=None, type=str,
                        dest='data.merge_dir', help='The Merge Label Directory of the data.')
    parser.add_argument('--max_thre', default=0.5, type=float,
                        dest='merge.max_thre', help='The max threshold for merging.')
    parser.add_argument('--min_thre', default=0.3, type=float,
                        dest='merge.min_thre', help='The min threshold for merging.')
    parser.add_argument('--round_decay', default=0.05, type=float,
                        dest='merge.round_decay', help='The round decay for merging.')
    parser.add_argument('--fscore_ratio', default=0.5, type=float,
                        dest='merge.fscore_ratio', help='The F-score ratio for merging.')
    parser.add_argument('--max_round', default=10, type=int,
                        dest='merge.max_round', help='The max round for merging.')
    parser.add_argument('--cm_test_set', default="test", type=str,
                        dest='merge.cm_test_set', help='The test directory of merging.')
    parser.add_argument('--cnt_thre', default=10, type=int,
                        dest='merge.cnt_thre', help='The merge cnt threshold for merging.')
    parser.add_argument('--acc_thre', default=0.01, type=float,
                        dest='merge.acc_thre', help='The acc threshold for merging.')
    parser.add_argument('--enable_fscore', type=str2bool, nargs='?', default=False,
                        dest='merge.enable_fscore', help='Whether to enable Fscore for filtering.')

    # ***********  Params for env.  **********
    parser.add_argument('--seed', default=None, type=int, help='manual seed')
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist', type=str2bool, nargs='?', default=False,
                        dest='distributed', help='Use CUDNN.')

    args_parser = parser.parse_args()

    if args_parser.seed is not None:
        random.seed(args_parser.seed + args_parser.local_rank)
        torch.manual_seed(args_parser.seed + args_parser.local_rank)
        if args_parser.gpu is not None:
            torch.cuda.manual_seed_all(args_parser.seed + args_parser.local_rank)

    configer = Configer(args_parser=args_parser)
    cudnn.enabled = True
    if configer.get('data', 'multiscale') is None: 
        cudnn.benchmark = args_parser.cudnn
    else:
        cudnn.benchmark = False

    if configer.get('gpu') is not None and not configer.get('distributed', default=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in configer.get('gpu'))

    if configer.get('network', 'norm_type') is None:
        configer.update('network.norm_type', 'batchnorm')

    if torch.cuda.device_count() <= 1 or configer.get('distributed', default=False):
        configer.update('network.gather', True)

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add('project_dir', project_dir)
    if configer.get('phase') == 'test':
        from tools.data_generator import DataGenerator
        DataGenerator.gen_toyset(project_dir)
        for source in range(configer.get('data', 'num_data_sources')):
            configer.update('data.src{}_label_path'.format(source), os.path.join(project_dir, 'toyset/label.txt'))

    configer.update('logging.logfile_level', None)

    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'),
             dist_rank=configer.get('local_rank'))

    Log.info('BN Type is {}.'.format(configer.get('network', 'norm_type')))
    Log.info('Config Dict: {}'.format(json.dumps(configer.to_dict(), indent=2)))
    torch.cuda.empty_cache()
    if configer.get('method') == 'auto_sku_merger':
        from lib.auto_sku_merger import AutoSKUMerger
        auto_sku_merger = AutoSKUMerger(configer)
        auto_sku_merger.run()
    elif configer.get('method') == 'image_classifier':
        from lib.image_classifier import ImageClassifier
        model_distiller = ImageClassifier(configer)
        model_distiller.run()
    elif configer.get('method') == 'multitask_classifier':
        from lib.multitask_classifier import MultiTaskClassifier
        multitask_distiller = MultiTaskClassifier(configer)
        multitask_distiller.run()
    else:
        Log.error('Invalid method: {}'.format(configer.get('method')))
        exit()
