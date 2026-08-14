import os
import torch
import argparse
from basicsr.models.archs.fftformer_arch import fftformer
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as transforms
from glob import glob

def run_deblurring(args):
    os.makedirs(args.result_dir, exist_ok=True)

    model = fftformer()
    checkpoint = torch.load(args.model_checkpoint)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    input_img = Image.open(args.input_image).convert('RGB')
    input_tensor = F.to_tensor(input_img).unsqueeze(0).to(device)
    
    b, c, h, w = input_tensor.shape
    h_pad = (32 - h % 32) % 32
    w_pad = (32 - w % 32) % 32
    input_tensor = torch.nn.functional.pad(input_tensor, (0, w_pad, 0, h_pad), mode='reflect')
    
    with torch.no_grad():
        pred = model(input_tensor)
        pred = pred[:, :, :h, :w]
        pred = torch.clamp(pred, 0, 1)

    pred_image = F.to_pil_image(pred.squeeze(0).cpu())
    input_name = os.path.splitext(os.path.basename(args.input_image))[0]
    save_path = os.path.join(args.result_dir, f'{input_name}_deblurred.png')
    pred_image.save(save_path)
    # print(f'Deblurred image saved to: {save_path}')


def get_first_image(input_dir):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    all_images = []
    for ext in exts:
        all_images.extend(glob(os.path.join(input_dir, ext)))
    if not all_images:
        raise FileNotFoundError(f"No image found in {input_dir}")
    return sorted(all_images)[0] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Image Deblurring")
    
    parser.add_argument('--input_dir', type=str, default="", help='Path to input blurry image')
    parser.add_argument('--model_checkpoint', type=str, default="", help='Path to model .pth checkpoint')
    parser.add_argument('--result_dir', type=str, default="", help='Directory to save output image')

    args = parser.parse_args()
    args.input_image = get_first_image(args.input_dir)
    
    run_deblurring(args)
