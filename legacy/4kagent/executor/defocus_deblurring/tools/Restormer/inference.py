## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import sys
import math
import gc
import argparse
import datetime
import collections
from glob import glob
from runpy import run_path
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from skimage import img_as_ubyte
from natsort import natsorted
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity


script_dir = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(description='Test Restormer on your own images')
parser.add_argument('--input_dir', default='./demo/degraded/', type=str, help='Input image directory or file')
parser.add_argument('--result_dir', default='./demo/restored/', type=str, help='Output directory for restored images')
parser.add_argument('--task', required=True, type=str, choices=[
    'Motion_Deblurring',
    'Single_Image_Defocus_Deblurring',
    'Deraining',
    'Real_Denoising',
    'Gaussian_Gray_Denoising',
    'Gaussian_Color_Denoising'
])
parser.add_argument('--tile', type=int, default=None, help='Tile size for large images (e.g. 720)')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlap between tiles')
parser.add_argument('--ckpt', type=str, default="")
args = parser.parse_args()


task = args.task
inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir, task)
os.makedirs(out_dir, exist_ok=True)


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)


def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] = 1
        parameters['out_channels'] = 1
        parameters['LayerNorm_type'] = 'BiasFree'
    else:
        raise ValueError(f"Unsupported task: {task}")
    return weights, parameters


extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.' + ext)))
    files = natsorted(files)

if len(files) == 0:
    raise FileNotFoundError(f"No input images found in {inp_dir}")


parameters = {
    'inp_channels': 3,
    'out_channels': 3,
    'dim': 48,
    'num_blocks': [4, 6, 6, 8],
    'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8],
    'ffn_expansion_factor': 2.66,
    'bias': False,
    'LayerNorm_type': 'WithBias',
    'dual_pixel_task': False
}
_, parameters = get_weights_and_parameters(task, parameters)
weights = args.ckpt

arch_path = f'{script_dir}/basicsr/models/archs/restormer_arch.py'
load_arch = run_path(arch_path)
model = load_arch['Restormer'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 8


print(f"\n ==> Running {task} with weights {weights}\n")

with torch.no_grad():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img = load_gray_img(file_) if task == 'Gaussian_Gray_Denoising' else load_img(file_)
        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

        # Padding
        height, width = input_.shape[2], input_.shape[3]
        H = math.ceil(height / img_multiple_of) * img_multiple_of
        W = math.ceil(width / img_multiple_of) * img_multiple_of
        padh = H - height
        padw = W - width
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        if args.tile is None:
            restored = model(input_)
        else:
            b, c, h, w = input_.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "Tile size must be a multiple of 8"
            stride = tile - args.tile_overlap

            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

            E = torch.zeros(b, c, h, w).type_as(input_)
            W_map = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model(patch)
                    out_mask = torch.ones_like(out_patch)
                    E[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_patch)
                    W_map[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(out_mask)

            restored = E.div_(W_map)

        # Crop & Convert
        restored = restored[:, :, :height, :width]
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
        restored = img_as_ubyte(restored)

        filename = os.path.splitext(os.path.basename(file_))[0]
        save_path = os.path.join(out_dir, f"{filename}.png")

        if task == 'Gaussian_Gray_Denoising':
            save_gray_img(save_path, restored)
        else:
            save_img(save_path, restored)

print(f"\nRestored images are saved at {out_dir}")
