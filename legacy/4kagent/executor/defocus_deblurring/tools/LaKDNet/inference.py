import os
import yaml
import lpips
from datetime import datetime
from pathlib import Path
from glob import glob
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

import torch
import torchvision.utils as vutils

from util.util import *
from models.LaKDNet import *
import argparse
import cv2


parser = argparse.ArgumentParser(description='Defocus or Motion Testing')
parser.add_argument('--task_type', type=str, default='Defocus', help='Defocus | Motion')
parser.add_argument('--input_dir', type=str, default='test_images')
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--train_dataset', type=str, default='realj')
parser.add_argument('--ckpt', type=str, default='./ckpts/Defocus/train_on_dpdd_l/train_on_dpdd_l.pth')
args = parser.parse_args()


# 'Defocus' or 'Motion'
task_type = args.task_type
script_dir = os.path.dirname(os.path.abspath(__file__))
agent_inference_yml = f'{script_dir}/options/agentic_restoration.yml'


with open(agent_inference_yml, 'r') as file:
    config = yaml.safe_load(file).get(task_type, {})


if args.task_type == "Defocus":
    test_status = ['train_on_dpdd_l']
    net_configs = ['network_l']
elif args.task_type == 'Motion':
    if args.train_dataset == "gopro":
        test_status = ['train_on_gopro_l']
        net_configs = ['network_l']
    if args.train_dataset == "realj":
        test_status = ['train_on_realj_l']
        net_configs = ['network_l']
    if args.train_dataset == "realr":
        test_status = ['train_on_realr_l']
        net_configs = ['network_l']

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

for idx, test_element in enumerate(test_status):
    print(f'Test {idx}: {test_element}')

    if idx >= len(net_configs):
        print(f"Warning: Index {idx} out of net_configs range. Skipping...")
        continue

    net_config = config.get(net_configs[idx], {})
    net_weight = args.ckpt

    if net_weight is None:
        print(f"Warning: No weight found for {test_element}. Skipping...")
        continue

    net_dual = 'dual' in test_element

    image_list = sorted(os.listdir(args.input_dir))
    if not image_list:
        print(f"Error: No images found in {args.input_dir}.")
        continue

    image_path = os.path.join(args.input_dir, image_list[0])
    filename = os.path.basename(image_path)

    if not net_dual:
        C = read_image(image_path, 255.0)
        C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).cuda()
        C, h, w = crop_image(C, 8, True)

    with torch.no_grad():
        network = LaKDNet(**net_config).cuda()
        network.load_state_dict(torch.load(net_weight))
        
        if not net_dual:
            output = network(C)

    output = output[:, :, :h, :w]
    output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)
    
    output_image_bgr = cv2.cvtColor(output_cpu, cv2.COLOR_RGB2BGR)

    save_file_path_deblur = os.path.join(output_dir, filename)
    print(f"Saving output to {save_file_path_deblur}")
    
    output_image_bgr = (output_image_bgr * 255).astype('uint8')
    cv2.imwrite(save_file_path_deblur, output_image_bgr)

