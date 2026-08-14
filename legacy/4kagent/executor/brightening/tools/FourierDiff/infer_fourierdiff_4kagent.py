import argparse
import logging
import os
import sys
import traceback
import yaml
import torch
import numpy as np

from guided_diffusion.diffusion_inference import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument("--config", type=str, default="./configs/4kagent.yml", help="Path to the config file")
    parser.add_argument("-i", "--output_dir", type=str, default="", help="Output directory for samples")
    parser.add_argument("--path_y", type=str, default="", help="Path of the test dataset")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility")
    parser.add_argument("--exp", type=str, default="exp", help="Experiment path for saving related data")
    parser.add_argument("--sigma_y", type=float, default=0.0, help="Sigma_y parameter")
    parser.add_argument("--eta", type=float, default=0.85, help="Eta parameter")
    parser.add_argument("--verbose", type=str, default="info", help="Logging level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_true", help="No interaction mode (suitable for Slurm jobs)")
    parser.add_argument("--subset_start", type=int, default=-1, help="Start index for subset processing")
    parser.add_argument("--subset_end", type=int, default=-1, help="End index for subset processing")
    parser.add_argument("-n", "--noise_type", type=str, default="gaussian", choices=["gaussian", "3d_gaussian", "poisson", "speckle"], help="Type of noise to add")
    parser.add_argument("--add_noise", action="store_true", help="Enable noise addition")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    logging_level = getattr(logging, args.verbose.upper(), logging.INFO)
    logging.basicConfig(
        level=logging_level,
        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    new_config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    """Convert dictionary to argparse.Namespace recursively"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def main():
    args, config = parse_args_and_config()

    try:
        runner = Diffusion(args, config)
        runner.sample()
    except Exception as e:
        logging.error("An error occurred:\n" + traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
