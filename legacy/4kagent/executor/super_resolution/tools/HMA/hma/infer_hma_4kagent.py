# modified from https://github.com/XPixelGroup/HAT/blob/main/hat/test.py
# flake8: noqa

import argparse
import logging
import os.path as osp
import random
import yaml
import torch
from collections import OrderedDict

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

import hma.archs
import hma.data
import hma.models

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.models.sr_model import SRModel
from basicsr.test import test_pipeline
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs, set_random_seed
from basicsr.utils.options import dict2str, parse_options, ordered_yaml, _postprocess_yml_value

from basicsr.utils.dist_util import get_dist_info, init_dist


def custom_parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--force_yml', nargs='+', default=None, help='Force update YAML keys. Example: train:ema_decay=0.999')
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        init_dist(args.launcher, **opt.get('dist_params', {}))

    opt['rank'], opt['world_size'] = get_dist_info()

    seed = opt.get('manual_seed', random.randint(1, 10000))
    opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.force_yml:
        for entry in args.force_yml:
            keys, value = entry.split('=')
            value = _postprocess_yml_value(value.strip())
            eval_str = 'opt'
            for key in keys.strip().split(':'):
                eval_str += f'["{key}"]'
            exec(eval_str + '=value')

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        for k in ['dataroot_gt', 'dataroot_lq']:
            if dataset.get(k) is not None:
                dataset[k] = osp.expanduser(dataset[k])

    for k, v in opt['path'].items():
        if v and ('resume_state' in k or 'pretrain_network' in k):
            opt['path'][k] = osp.expanduser(v)

    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path'].update({
            'experiments_root': experiments_root,
            'models': osp.join(experiments_root, 'models'),
            'training_states': osp.join(experiments_root, 'training_states'),
            'log': experiments_root,
            'visualization': osp.join(experiments_root, 'visualization'),
        })

        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger'].update({'print_freq': 1, 'save_checkpoint_freq': 8})
    else:
        # path settings for test
        results_root = osp.join(opt['path']['results'], opt['name'])
        opt['path'].update({
            'results_root': results_root,
            'log': results_root,
            'visualization': osp.join(results_root, 'visualization'),
        })

    return opt, args


def custom_test_pipeline(root_path):
    opt, _ = custom_parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger('basicsr', logging.INFO, log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'],
            sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    model: SRModel = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    custom_test_pipeline(root_path)
