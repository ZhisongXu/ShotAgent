import os
import torch
import argparse
from glob import glob
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn.functional as Func
from models.EVSSM import EVSSM


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return F.to_tensor(img).unsqueeze(0)


def save_image(tensor, save_path):
    tensor = torch.clamp(tensor, 0, 1) + 0.5 / 255
    img = F.to_pil_image(tensor.squeeze(0).cpu(), 'RGB')
    img.save(save_path)
    

def run_deblurring(args):
    model = EVSSM()
    state_dict = torch.load(args.ckpts)['params']
    model.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # Prepare result directory
    os.makedirs(args.save_image_dir, exist_ok=True)

    # Load and process image
    input_tensor = load_image(args.input_image).to(device)
    _, _, h, w = input_tensor.size()

    # Compute padding to make h, w divisible by 4
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4

    # Pad (pad_left=0, pad_right=pad_w, pad_top=0, pad_bottom=pad_h)
    padded = Func.pad(input_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # Inference
    with torch.no_grad():
        pred_padded = model(padded)

    # Crop back to original size
    pred = pred_padded[..., :h, :w]

    # Save result
    filename = os.path.basename(args.input_image)
    save_path = os.path.join(args.save_image_dir, filename)
    save_image(pred, save_path)

def get_first_image(input_dir):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    all_images = []
    for ext in exts:
        all_images.extend(glob(os.path.join(input_dir, ext)))
    if not all_images:
        raise FileNotFoundError(f"No image found in {input_dir}")
    return sorted(all_images)[0] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Image Deblurring with EVSSM")

    parser.add_argument('--input_dir', type=str, default="", help='Path to input blurry image')
    parser.add_argument('--ckpts', type=str, default="", help='Path to model .pth checkpoint')
    parser.add_argument('--save_image_dir', type=str, default="", help='Directory to save output image')

    args = parser.parse_args()
    args.input_image = get_first_image(args.input_dir)
    
    run_deblurring(args)
