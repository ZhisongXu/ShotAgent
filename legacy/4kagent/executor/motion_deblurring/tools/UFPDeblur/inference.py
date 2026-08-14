import argparse
import torch
import os
from PIL import Image
from glob import glob

from torchvision.transforms.functional import to_tensor, to_pil_image

from basicsr.models.archs.UFPNet_code_uncertainty_arch import UFPNet_code_uncertainty_Local


def load_model(model_path):
    print("Loading model...")
    model = UFPNet_code_uncertainty_Local(
        img_channel=3,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
        kernel_size=19,
        train_size=(1, 3, 256, 256),
        fast_imp=False
    )
    model.load_state_dict(torch.load(model_path)["params"], strict=True)
    model.eval().cuda()
    return model


def run_deblurring(model, input_path, result_dir):
    img = Image.open(input_path).convert("RGB")
    img_tensor = to_tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        kernel, output = model(img_tensor)
        output = torch.clamp(output, 0, 1)

    output_image = to_pil_image(output.squeeze(0).cpu())
    
    os.makedirs(result_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    save_path = os.path.join(result_dir, filename)
    output_image.save(save_path)
    print(f"Saved deblurred image to: {save_path}")


def get_first_image(input_dir):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    all_images = []
    for ext in exts:
        all_images.extend(glob(os.path.join(input_dir, ext)))
    if not all_images:
        raise FileNotFoundError(f"No image found in {input_dir}")
    return sorted(all_images)[0] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Image Inference for UFPNet_code_uncertainty_Local")
    # parser.add_argument('--input_image', type=str, default="./input/motion deblurring.png", help='Path to input blurry image')
    parser.add_argument('--model_checkpoint', type=str, default="./UFPDeblur/pretrain_models/train_on_GoPro/net_g_latest.pth", help='Path to .pth model checkpoint')
    parser.add_argument('--input_dir', type=str, default="./input", help='Path to input blurry image')
    parser.add_argument('--result_dir', type=str, default="./UFPDeblur/results", help='Directory to save output image')

    args = parser.parse_args()
    args.input_image = get_first_image(args.input_dir)
    
    model = load_model(args.model_checkpoint)
    
    run_deblurring(model, args.input_image, args.result_dir)
