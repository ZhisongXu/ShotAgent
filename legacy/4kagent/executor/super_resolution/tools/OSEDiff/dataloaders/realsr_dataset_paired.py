import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from dataloaders.realesrgan import RealESRGAN_degradation

import cv2

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list = []
            self.lq_list = []

            assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.dataset_txt_paths_list)):
                with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')

                # LQ dataset list
                if args.dataset_lq_txt_paths_list:
                    with open(args.dataset_lq_txt_paths_list[idx_dataset], 'r') as f:
                        lq_dataset_list = [line.strip() for line in f.readlines()]
                        self.lq_list += lq_dataset_list
                else:
                    raise ValueError('LQ dataset list is not provided.')
                    # self.lq_list = [None] * len(self.gt_list)
        
        elif split == 'test':
            self.gt_list = []
            self.lq_list = []

            assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.dataset_txt_paths_list)):
                with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    self.gt_list += dataset_list

                # LQ dataset list
                if args.dataset_lq_txt_paths_list:
                    with open(args.dataset_lq_txt_paths_list[idx_dataset], 'r') as f:
                        lq_dataset_list = [line.strip() for line in f.readlines()]
                        self.lq_list += lq_dataset_list
                else:
                    raise ValueError('LQ dataset list is not provided.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            # gt_img = self.crop_preproc(gt_img)

            if self.lq_list[idx]:
                lq_img = Image.open(self.lq_list[idx]).convert('RGB')
                # lq_img = self.crop_preproc(lq_img)
            else:
                raise ValueError('LQ dataset list is not provided.')
                # _, lq_img = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
                # lq_img = lq_img.squeeze(0)

            # output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            # output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            
            # gt_img, lq_img = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            gt_img = img2tensor([np.asarray(gt_img)/255.], bgr2rgb=False, float32=True)[0].unsqueeze(0).to('cpu')
            lq_img = img2tensor([np.asarray(lq_img)/255.], bgr2rgb=False, float32=True)[0].unsqueeze(0).to('cpu')

            # images scaled to -1,1
            gt_img = F.normalize(gt_img.squeeze(0), mean=[0.5], std=[0.5])
            lq_img = F.normalize(lq_img.squeeze(0), mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["output_pixel_values"] = gt_img
            example["conditioning_pixel_values"] = lq_img

            return example