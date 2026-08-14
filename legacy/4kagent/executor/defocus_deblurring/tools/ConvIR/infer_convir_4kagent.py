import os
import torch
import argparse
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from Motion_Deblurring.models.ConvIR import build_net
from glob import glob

factor = 32

def pad_image(img_tensor):
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    img_tensor = F.pad(img_tensor, (0, padw, 0, padh), mode='reflect')
    return img_tensor, h, w

def run_deblurring(model, input_path, output_dir, device):
    # Load image
    img = Image.open(input_path).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    # Pad
    img_tensor, orig_h, orig_w = pad_image(img_tensor)

    # Inference
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)[2]
        pred = pred[:, :, :orig_h, :orig_w]
        pred = torch.clamp(pred, 0, 1)

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    save_path = os.path.join(output_dir, filename)
    ToPILImage()(pred.squeeze(0).cpu()).save(save_path)
    print(f"Saved output to: {save_path}")


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
    parser.add_argument('--input_dir', type=str, default="./ConvIR/example", help='Path to input blurry image')
    parser.add_argument('--model_checkpoint', type=str, default="./ConvIR/pretrain_models/dpdd-large.pkl", help='Path to model .pkl checkpoint')
    parser.add_argument('--result_dir', type=str, default="./ConvIR/results", help='Directory to save output image')
    
    args = parser.parse_args()
    args.input_image = get_first_image(args.input_dir)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_net().to(device)
    state_dict = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(state_dict['model'])

    # Run inference
    run_deblurring(model, args.input_image, args.result_dir, device)
