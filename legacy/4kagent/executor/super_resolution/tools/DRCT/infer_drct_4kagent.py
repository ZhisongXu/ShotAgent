import argparse
import os
import glob
import cv2
import numpy as np
import torch

from drct.archs.DRCT_arch import DRCT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/work/u1657859/DRCT/experiments/train_DRCT-L_SRx4_finetune_from_ImageNet_pretrain/models/DRCT-L.pth")
    parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx4', help='Input folder with LR images')
    parser.add_argument('--output', type=str, default='results/DRCT-L', help='Output folder for SR images')
    parser.add_argument('--scale', type=int, default=4, help='Upscale factor')
    parser.add_argument('--tile', type=int, default=512, help='Tile size for memory-efficient testing')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlap size for tile inference')
    return parser.parse_args()

def load_model(model_path, device):
    model = DRCT(
        upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
        squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
        depths=[6]*12, embed_dim=180, num_heads=[6]*12, gc=32,
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.to(device).eval()
    return model

def preprocess_image(path, device):
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).float().to(device)
    return img

def pad_image(img, window_size):
    b, c, h, w = img.shape
    h_pad = (h // window_size + 1) * window_size - h
    w_pad = (w // window_size + 1) * window_size - w
    img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + h_pad, :]
    img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + w_pad]
    return img, h, w

def test(img_lq, model, args, window_size):
    if args.tile is None:
        return model(img_lq)
    
    # tiled inference
    b, c, h, w = img_lq.size()
    tile = min(args.tile, h, w)
    stride = tile - args.tile_overlap
    sf = args.scale
    assert tile % window_size == 0, "Tile size should be a multiple of window size"

    E = torch.zeros(b, c, h * sf, w * sf).to(img_lq)
    W = torch.zeros_like(E)

    for y in list(range(0, h - tile, stride)) + [h - tile]:
        for x in list(range(0, w - tile, stride)) + [w - tile]:
            patch = img_lq[..., y:y+tile, x:x+tile]
            out_patch = model(patch)
            E[..., y*sf:(y+tile)*sf, x*sf:(x+tile)*sf] += out_patch
            W[..., y*sf:(y+tile)*sf, x*sf:(x+tile)*sf] += 1

    return E / W

def save_image(tensor, path):
    output = tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # RGB
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(path, output)

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_path, device)
    window_size = 16

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print(f'Testing [{idx}]: {imgname}')

        try:
            img = preprocess_image(path, device)
            img, h_old, w_old = pad_image(img, window_size)
            output = test(img, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
        except Exception as e:
            print(f'Error processing {imgname}: {e}')
            continue

        save_path = os.path.join(args.output, f'{imgname}_DRCT-L_X{args.scale}.png')
        save_image(output, save_path)

if __name__ == '__main__':
    main()
