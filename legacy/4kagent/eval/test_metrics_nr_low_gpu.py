# Image Quality Assessment Script (Memory-Optimized)
# Evaluates metrics like CLIPIQA, NIQE, MUSIQ, MANIQA sequentially to limit GPU memory usage.

import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import cv2
import numpy as np
import torch

import pyiqa
from basicsr.utils import img2tensor

def get_timestamp():
    """Returns the current timestamp in a specific format."""
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def dict2str(opt, indent=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += ' ' * (indent * 2) + "]\n"
        else:
            msg += ' ' * (indent * 2) + f"{k}: {v}\n"
    return msg


def evaluate_metric(metric_name, metric, img_paths, args, logger, device):
    """
    Evaluate a single IQA metric over a list of image paths.
    Returns the average metric score.
    """
    total = 0.0
    for sr_path in img_paths:
        sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
        if sr_img is None:
            logger.warning(f"Failed to read {sr_path}, skipping.")
            continue
        if args.crop_border > 0:
            sr_img = sr_img[args.crop_border:-args.crop_border,
                            args.crop_border:-args.crop_border, ...]
        sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0)
        sr_tensor = sr_tensor.to(device).contiguous() / 255.0
        with torch.no_grad():
            score = metric(sr_tensor).item()
        total += score
    avg = total / len(img_paths) if img_paths else 0.0
    return avg


def main():
    parser = argparse.ArgumentParser(description="Memory-Optimized IQA Script")
    parser.add_argument("--inp_imgs", nargs='+', required=True,
                        help="Paths to input (SR) image directories.")
    parser.add_argument("--log", type=str, required=True,
                        help="Directory to save logs.")
    parser.add_argument("--log_name", type=str, default='METRICS',
                        help="Base name for the log files.")
    parser.add_argument("--crop_border", type=int, default=0,
                        help="Crop border for PSNR/SSIM cropping.")
    args = parser.parse_args()

    # Device selection: prefer GPU, otherwise CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Prepare logging
    os.makedirs(args.log, exist_ok=True)

    # Initialize logger
    # Assuming the first init image path has enough parts
    try:
        args.log_name = args.inp_imgs[0].split('/')[8]
    except IndexError:
        args.log_name = 'METRICS'
    setup_logger('base', args.log, f'test_{args.log_name}', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info("===== Configuration =====")
    logger.info(dict2str(vars(args)))
    logger.info("=========================")

    # Define list of IQA metrics to evaluate sequentially
    metric_names = ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA']
    metric_list = ['clipiqa', 'niqe', 'musiq', 'maniqa-pipal']

    # Collect all directories
    all_dirs = [os.path.normpath(d) for d in args.inp_imgs]

    logger.info(f"Starting evaluation for {len(all_dirs)} directories.")

    init_imgs_names = []
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))

        dir_name = os.path.basename(os.path.normpath(init_dir))
        init_imgs_names.append(dir_name)

        logger.info(f"Directory [{dir_name}]: {len(img_sr_list)} SR images.")

    logger.info("\n===== Starting Evaluation =====\n")

    # Iterate over each directory
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))
        dir_name = init_imgs_names[dir_idx]

        # Initialize accumulators for average metrics
        metrics_accum = {metric: 0.0 for metric in metric_names}

        logger.info(f"Testing Directory: [{dir_name}]")

        # Iterate over each image pair
        for img_idx, sr_path in enumerate(img_sr_list):
            img_name = os.path.basename(sr_path)

            start_time = time.time()

            # Read and preprocess images
            sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)

            if args.crop_border > 0:
                sr_img = sr_img[args.crop_border:-args.crop_border, args.crop_border:-args.crop_border, ...]

            if sr_img is None:
                logger.warning(f"Image read failed for {img_name}. Skipping.")
                continue

            sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

            metrics = {}
            for name, metric_name in zip(metric_names, metric_list):
                # Load one metric at a time
                # logger.info(f"Loading metric: {metric_name}")
                # print(f"Loading metric: {metric_name}")
                metric = pyiqa.create_metric(metric_name, device=device)
                # Evaluate and record average
                metrics[name] = metric(sr_tensor).item()
                # Release GPU memory
                del metric
                torch.cuda.empty_cache()

            # Accumulate metrics
            for name in metrics_accum:
                metrics_accum[name] += metrics[name]

            # Calculate runtime
            end_time = time.time()
            runtime = end_time - start_time

            # Log per-image metrics and runtime
            metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            logger.info(f"{dir_name}/{img_name} | {metrics_str} | Runtime: {runtime:.2f} sec")

        # Compute average metrics
        num_images = len(img_sr_list)
        avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}

        # Log average metrics for the directory
        avg_metrics_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger.info(f"\n===== Average Metrics for [{dir_name}] =====\n{avg_metrics_str} \n")

        # Optionally, you can accumulate FID if needed for overall statistics

    logger.info("===== Evaluation Completed =====")


if __name__ == '__main__':
    main()
