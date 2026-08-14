import os
import gc
import sys
import glob
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from natsort import natsorted

sys.path.append(os.getcwd())

from osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from ram.models.ram_lora import ram
from ram import inference_ram as inference


tensor_transforms = transforms.ToTensor()
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def get_validation_prompt(args, image: Image.Image, model: torch.nn.Module, weight_dtype: torch.dtype) -> tuple[str, torch.Tensor]:
    device = "cuda"
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    return f"{captions[0]}, {args.prompt},", lq


def prepare_image(image: Image.Image, args) -> tuple[Image.Image, bool, int, int]:
    ori_width, ori_height = image.size
    rscale = args.upscale
    resize_flag = False

    if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
        scale = (args.process_size // rscale) / min(ori_width, ori_height)
        image = image.resize((int(scale * ori_width), int(scale * ori_height)))
        resize_flag = True

    image = image.resize((image.size[0] * rscale, image.size[1] * rscale))

    if image.width % 8 != 0 or image.height % 8 != 0:
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        resize_flag = True

    return image, resize_flag, ori_width, ori_height


def apply_alignment(output_pil: Image.Image, input_image: Image.Image, args) -> Image.Image:
    if args.align_method == 'adain':
        return adain_color_fix(target=output_pil, source=input_image)
    elif args.align_method == 'wavelet':
        return wavelet_color_fix(target=output_pil, source=input_image)
    else:
        return output_pil


def save_prompt_txt(prompt: str, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.txt"), 'w', encoding='utf-8') as f:
        f.write(prompt)


def inference_one_image(image_path: str, model, ram_model, args, weight_dtype):
    input_image = Image.open(image_path).convert('RGB')
    input_image, resize_flag, ori_w, ori_h = prepare_image(input_image, args)

    basename = os.path.basename(image_path)
    prompt, lq_tensor = get_validation_prompt(args, input_image, ram_model, weight_dtype)

    if args.save_prompts:
        save_prompt_txt(prompt, os.path.splitext(basename)[0], os.path.join(args.output_dir, 'txt'))

    print(f"Processing {image_path}, tag: {prompt}".encode('utf-8'))

    with torch.no_grad():
        lq_tensor = lq_tensor * 2 - 1
        output = model(lq_tensor, prompt=prompt)
        output_image = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_image = apply_alignment(output_image, input_image, args)

        if resize_flag:
            output_image = output_image.resize((args.upscale * ori_w, args.upscale * ori_h))

        output_path = os.path.join(args.output_dir, basename)
        output_image.save(output_path)

    flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--osediff_path", type=str, default='preset/models/osediff.pkl')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--save_prompts', type=bool, default=False)
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    parser.add_argument("--merge_and_unload_lora", default=False)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    args = parser.parse_args()

    model = OSEDiff_test(args)

    if os.path.isdir(args.input_image):
        image_paths = natsorted(glob.glob(f'{args.input_image}/*.[pPbBjJ][nNmMgGpP]*') + glob.glob(f'{args.input_image}/*.raw'))
    else:
        image_paths = [args.input_image]

    print(f'Total images: {len(image_paths)}')

    ram_model = ram(pretrained=args.ram_path,
                    pretrained_condition=args.ram_ft_path,
                    image_size=384,
                    vit='swin_l').eval().to("cuda")

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    ram_model = ram_model.to(dtype=weight_dtype)
    
    os.makedirs(args.output_dir, exist_ok=True)

    for image_path in image_paths:
        inference_one_image(image_path, model, ram_model, args, weight_dtype)
        print('-----------------------------------------------')


if __name__ == "__main__":
    main()
