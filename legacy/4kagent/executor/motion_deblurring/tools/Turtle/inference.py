import argparse
import os
import time
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from importlib import import_module
from basicsr.utils.options import parse

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor

def save_image(tensor, save_path):
    tensor = torch.clamp(tensor.squeeze(0), 0, 1).permute(1, 2, 0).cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, tensor)

def create_model(args, model_type="t1"):
    if model_type == "t0":
        module = import_module('basicsr.models.archs.turtle_arch')
    elif model_type == "t1":
        module = import_module('basicsr.models.archs.turtle_t1_arch')
    elif model_type == "SR":
        module = import_module('basicsr.models.archs.turtle_super_t1_arch')
    else:
        raise ValueError("Unsupported model type")
    return module.make_model(args)

def run_deblurring(args):
    model_args = parse(args.config, is_train=True)
    model = create_model(model_args, model_type="t1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(ckpt['params'])
    model = model.to(device).eval()
    print(f"[INFO] Model loaded from {args.model_checkpoint}")

    
    image_list = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    if not image_list:
        raise FileNotFoundError(f"No images found in {args.input_dir}")
    input_image_path = image_list[0]
    print(f"[INFO] Using image: {input_image_path}")

    
    img = load_image(input_image_path).to(device)  # [1, 3, H, W]
    _, _, h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

    
    with torch.no_grad():
        x = torch.cat([img_pad, img_pad], dim=0).unsqueeze(0).to(device)  # [1, 6, H, W]
        output, _, _ = model(x, None, None)
        output = output[..., :h, :w]  # remove padding

    
    os.makedirs(args.result_dir, exist_ok=True)
    save_path = os.path.join(args.result_dir, os.path.basename(input_image_path))
    save_image(output, save_path)
    print(f"[RESULT] Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./Turtle/example", help='Directory containing one blurry image')
    parser.add_argument('--model_checkpoint', type=str, default="./Turtle/pretrain_models/GoPro_Deblur.pth", help='Path to .pth model checkpoint')
    parser.add_argument('--result_dir', type=str, default="./Turtle/results", help='Directory to save output')
    parser.add_argument('--config', type=str, default="./Turtle/options/Turtle_Deblur_Gopro.yml", help='Path to model config YAML file')
    args = parser.parse_args()

    start = time.time()
    run_deblurring(args)
    print(f"Finished in {time.time() - start:.2f}s")
