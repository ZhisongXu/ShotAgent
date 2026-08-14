# Modified from https://github.com/XPixelGroup/HAT/blob/main/hat/test.py
# flake8: noqa

import os.path as osp
import argparse
import logging
import random
import yaml
import torch
from collections import OrderedDict

# Register models and data
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

import hat.archs
import hat.data
import hat.models

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs, set_random_seed
from basicsr.utils.options import dict2str, parse_options, ordered_yaml, _postprocess_yml_value
from basicsr.utils.dist_util import get_dist_info, init_dist


def custom_parse_options(root_path, is_train=True):
    """Custom YAML + CLI parsing for BasicSR."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--force_yml', nargs='+', default=None,
                        help='Force update yml. Example: train:ema_decay=0.999')
    args = parser.parse_args()

    # Parse YAML
    with open(args.opt, 'r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # Distributed setup
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # Random seed
    seed = opt.get('manual_seed', random.randint(1, 10000))
    opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # Force update YAML values
    if args.force_yml is not None:
        for entry in args.force_yml:
            keys, value = map(str.strip, entry.split('='))
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for k in keys.split(':'):
                eval_str += f'["{k}"]'
            eval_str += '=value'
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # Debug mode patch
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt.get('num_gpu') == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # Expand dataset paths
    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase.split('_')[0]
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt'):
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq'):
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # Expand pretrain/resume paths
    for key, val in opt['path'].items():
        if val and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    # Path setup
    if is_train:
        exp_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path'].update({
            'experiments_root': exp_root,
            'models': osp.join(exp_root, 'models'),
            'training_states': osp.join(exp_root, 'training_states'),
            'log': exp_root,
            'visualization': osp.join(exp_root, 'visualization')
        })
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:
        # ⬇️ Your custom results_root logic preserved
        results_root = osp.join(opt['path']['results'], opt['name'])
        opt['path'].update({
            'results_root': results_root,
            'log': results_root,
            'visualization': osp.join(results_root, 'visualization')
        })

    return opt, args


def custom_test_pipeline(root_path):
    """Custom test pipeline using BasicSR."""
    opt, _ = custom_parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # Build test datasets
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed']
        )
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # Build model
    model: SRModel = build_model(opt)

    # Run validation
    for test_loader in test_loaders:
        name = test_loader.dataset.opt['name']
        logger.info(f'Testing {name}...')
        model.validation(test_loader, current_iter=opt['name'],
                         tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    custom_test_pipeline(root_path)
