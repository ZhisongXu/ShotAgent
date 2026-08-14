import argparse
import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from basicsr.models.archs.MLWNet_arch import MLWNet_Local


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img


def save_img(img_tensor, save_path):
    img = torch.clamp(img_tensor, 0, 1).cpu().permute(1, 2, 0).numpy()  # CHW -> HWC
    img = img_as_ubyte(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['params'], strict=True)
    print(f"[INFO] Loaded model checkpoint from: {ckpt_path}")
    return model


def run_deblurring(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = load_img(args.input_image)
    h, w = img.shape[:2]
    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)  # 1x3xHxW

    factor = 16
    H = ((h + factor - 1) // factor) * factor
    W = ((w + factor - 1) // factor) * factor
    padh = H - h
    padw = W - w
    img_tensor = F.pad(img_tensor, (0, padw, 0, padh), mode='reflect')

    model = MLWNet_Local(dim=64, base_size=int(256 * 1.5)).to(device)
    model = load_checkpoint(model, args.model_checkpoint, device)
    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = output[..., :h, :w].squeeze(0)  # remove batch dim
    
    os.makedirs(args.result_dir, exist_ok=True)
    save_path = os.path.join(args.result_dir, os.path.basename(args.input_image))
    save_img(output, save_path)
    print(f"Output saved to {save_path}")


def get_first_image(input_dir):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    all_images = []
    for ext in exts:
        all_images.extend(glob(os.path.join(input_dir, ext)))
    if not all_images:
        raise FileNotFoundError(f"No image found in {input_dir}")
    return sorted(all_images)[0] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_image', type=str, default="./input/motion deblurring.png", help='Path to input blurry image')
    parser.add_argument('--input_dir', type=str, default="./input", help='Path to input blurry image')
    parser.add_argument('--model_checkpoint', type=str, default="./MLWNet/pretrain_models/gopro-width64.pth", help='Path to model .pth checkpoint')
    parser.add_argument('--result_dir', type=str, default="./MLWNet/results", help='Directory to save output image')

    args = parser.parse_args()
    args.input_image = get_first_image(args.input_dir)
    
    run_deblurring(args)
