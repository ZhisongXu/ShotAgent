import os
import yaml
import torch
import argparse
import cv2
from pathlib import Path
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from util.util import *
from models.LaKDNet import *

def parse_args():
    parser = argparse.ArgumentParser(description='Defocus or Motion Testing')
    parser.add_argument('--task_type', type=str, default='Defocus', help='Defocus | Motion')
    parser.add_argument('--input_dir', type=str, default='train_val_images')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--train_dataset', type=str, default='realj')
    parser.add_argument('--ckpt', type=str, default='')
    return parser.parse_args()

def load_config(task_type):
    with open('./options/agentir_inference.yml', 'r') as file:
        return yaml.safe_load(file).get(task_type, {})

def get_test_settings(args):
    if args.task_type == "Defocus":
        return ['train_on_dpdd_l'], ['network_l'], args.ckpt or './ckpts/Defocus/train_on_dpdd_l/train_on_dpdd_l.pth'
    elif args.task_type == 'Motion':
        dataset_ckpt = {
            "gopro": './ckpts/Motion/train_on_gopro_l/train_on_gopro_l.pth',
            "realj": './LaKDNet/ckpts/Motion/train_on_realj_l/train_on_realj_l.pth',
            "realr": './LaKDNet/ckpts/Motion/train_on_realr_l/train_on_realr_l.pth'
        }
        return [f'train_on_{args.train_dataset}_l'], ['network_l'], dataset_ckpt.get(args.train_dataset, '')
    return [], [], ''

def process_images(network, input_subdir, output_subdir):
    os.makedirs(output_subdir, exist_ok=True)
    image_list = sorted(glob(os.path.join(input_subdir, '*.png')) + glob(os.path.join(input_subdir, '*.jpg')))
    if not image_list:
        print(f"Error: No images found in {input_subdir}.")
        return
    
    for image_path in image_list:
        filename = os.path.basename(image_path)
        C = read_image(image_path, 255.0)
        C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).cuda()
        C, h, w = crop_image(C, 8, True)
        
        with torch.no_grad():
            output = network(C)
        
        output = output[:, :, :h, :w].cpu().numpy()[0].transpose(1, 2, 0)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output_bgr = (output_bgr * 255).astype('uint8')
        
        save_path = os.path.join(output_subdir, filename)
        print(f"Saving output to {save_path}")
        cv2.imwrite(save_path, output_bgr)

def main():
    args = parse_args()
    config = load_config(args.task_type)
    test_status, net_configs, ckpt_path = get_test_settings(args)
    
    if not ckpt_path:
        print("Error: No valid checkpoint found.")
        return
    
    network = LaKDNet(**config.get(net_configs[0], {})).cuda()
    network.load_state_dict(torch.load(ckpt_path))
    
    input_subdirs = [d for d in Path(args.input_dir).iterdir() if d.is_dir()]
    
    for subdir in input_subdirs:
        output_subdir = Path(args.output_dir) / subdir.name
        process_images(network, str(subdir), str(output_subdir))
    
if __name__ == "__main__":
    main()
