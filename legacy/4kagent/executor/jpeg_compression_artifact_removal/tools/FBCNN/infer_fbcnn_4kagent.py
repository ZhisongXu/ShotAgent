# Modified from https://github.com/jiaxi-jiang/FBCNN/blob/main/main_test_fbcnn_color_real.py

import os
import argparse
import requests
import torch
import numpy as np
from utils import utils_image as util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--weight_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--qf', type=str)  # "blind", 5, or 90
    args = parser.parse_args()

    input_dir = args.input_dir
    weight_dir = args.weight_dir
    output_dir = args.output_dir
    qf = args.qf

    n_channels = 3
    model_name = 'fbcnn_color.pth'
    nc = [64, 128, 256, 512]
    nb = 4

    model_path = os.path.join(weight_dir, model_name)
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = f'https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{model_name}'
        print(f'Downloading model from {url}')
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    else:
        print(f'Loading model from {model_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_fbcnn import FBCNN as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad = False
    model = model.to(device)

    image_paths = util.get_image_paths(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img = util.imread_uint(img_path, n_channels=n_channels)
        img_tensor = util.uint2tensor4(img).to(device)

        if qf == 'blind':
            restored_img, pred_qf = model(img_tensor)
        else:
            qf_input = torch.tensor([[1 - int(qf) / 100.0]], device=device)
            restored_img, pred_qf = model(img_tensor, qf_input)

        restored_img = util.tensor2single(restored_img)
        restored_img = util.single2uint(restored_img)

        output_filename = os.path.join(output_dir, os.path.basename(img_path))
        util.imsave(restored_img, output_filename)

if __name__ == '__main__':
    main()
