import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage import img_as_ubyte
from collections import OrderedDict
from basicsr.models.archs.AdaRevID_arch import AdaRevIDSlide as Net
import Motion_Deblurring.utils as utils
import yaml
from glob import glob


def main(args):
    # Load model config
    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model_config = config['network_g']
    model_config.pop('type', None)
    model_config['state_dict_pth_classifier'] = args.state_dict_pth_classifier
    model_config['cal_num_decoders'] = False

    # Initialize model
    model = Net(**model_config)
    checkpoint = torch.load(args.model_checkpoint)

    try:
        model.load_state_dict(checkpoint['params'], strict=False)
    except:
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)

    model = model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    # Load and preprocess image
    img = np.float32(utils.load_img(args.input_image)) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

    # Inference
    with torch.no_grad():
        start = time.time()
        restored = model(img_tensor)
        end = time.time()

        if isinstance(restored, list):
            restored = restored[-1]
        elif isinstance(restored, dict):
            restored = restored.get('img', restored)
            if isinstance(restored, list):
                restored = restored[-1]

        restored = torch.clamp(restored, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored_img = img_as_ubyte(restored)

    # Save result
    os.makedirs(args.save_image_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(args.input_image))[0] + '_restored.png'
    save_path = os.path.join(args.save_image_dir, filename)
    utils.save_img(save_path, restored_img)

    print(f"Restored image saved to: {save_path}")
    print(f"Inference time: {end - start:.4f} seconds")

def get_first_image(input_dir):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    all_images = []
    for ext in exts:
        all_images.extend(glob(os.path.join(input_dir, ext)))
    if not all_images:
        raise FileNotFoundError(f"No image found in {input_dir}")
    return sorted(all_images)[0] 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with AdaRevID on a single image')
    parser.add_argument('--input_dir', type=str, default="./input", help='Path to input blurry image')
    parser.add_argument('--save_image_dir', type=str, default='./AdaRevD/results', help='Directory to save result')
    parser.add_argument('--model_checkpoint', type=str, default="./AdaRevD/pretrain_model/RevD-L_GoPro/net_g_GoPro.pth", help='Path to model weights')
    parser.add_argument('--state_dict_pth_classifier', type=str, default="./AdaRevD/pretrain_model/classifier/GoPro.pth", help='Path to model weights')
    parser.add_argument('--yaml_config', type=str, default="./AdaRevD/Motion_Deblurring/Options/AdaRevID-B-GoPro-test.yml", help='Path to yaml config')
    args = parser.parse_args()
    
    args.input_image = get_first_image(args.input_dir)
    
    main(args)