import os
import gc
import sys
import glob
import argparse

import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

from natsort import natsorted
from ram.models.ram_lora import ram
from ram import inference_ram as inference

from osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

sys.path.append(os.getcwd())
device = 'cuda'
weight_dtype = torch.float16  # Or change to torch.float32 if needed

# Transforms
tensor_transforms = transforms.ToTensor()
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def flush():
    """Clear memory cache."""
    gc.collect()
    torch.cuda.empty_cache()


def load_ram_model(ram_path, ram_ft_path, image_size=384, vit_type='swin_l'):
    model = ram(
        pretrained=ram_path,
        pretrained_condition=ram_ft_path,
        image_size=image_size,
        vit=vit_type
    )
    model.eval()
    model = model.to(device=device, dtype=weight_dtype)
    return model


def get_validation_prompt(image: Image.Image, model) -> tuple[str, torch.Tensor]:
    """Generate a prompt using RAM model."""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    print('captions:', captions)
    return captions[0], lq


def main(image_path: str):
    print(f"[INFO] Loading model...")
    ram_model = load_ram_model(
        ram_path='pretrained_models/RAM/ram_swin_large_14m.pth',
        ram_ft_path='pretrained_models/DAPE/DAPE.pth'
    )

    print(f"[INFO] Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    prompt, lq_tensor = get_validation_prompt(image, ram_model)

    print('validation_prompt:', prompt)
    flush()
    print('-----------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./dataset/RealSRSet/comic3.png",
                        help='Path to the input image (e.g., /path/to/image.png)')
    args = parser.parse_args()
    main(args.image_path)
