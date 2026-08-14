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

        self.gt_list = []
        self.lq_list = []

            
        for idx_dataset in range(len(args.dataset_txt_paths_list)):
            with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
                gt_dataset_list = [line.strip() for line in f.readlines()]

            with open(args.dataset_lq_txt_paths_list[idx_dataset], 'r') as f:
                lq_dataset_list = [line.strip() for line in f.readlines()]
                
            if len(gt_dataset_list) != len(lq_dataset_list):
                raise ValueError("Mismatch between ground truth and LQ dataset lengths.")

        n_total = len(gt_dataset_list)
        n_train = int(n_total * 0.98)
        if self.split == 'train':
            gt_split = gt_dataset_list[:n_train]
            lq_split = lq_dataset_list[:n_train]
            # Multiply training data by the given probability factor.
            self.gt_list += gt_split
            self.lq_list += lq_split
            print(f'=====> Appended {len(gt_split)} training samples from dataset {idx_dataset}.')
        elif self.split == 'test':
            gt_split = gt_dataset_list[n_train:]
            lq_split = lq_dataset_list[n_train:]
            self.gt_list += gt_split
            self.lq_list += lq_split
            print(f'=====> Appended {len(gt_split)} testing samples from dataset {idx_dataset}.')
        else:
            raise ValueError("Invalid split: choose either 'train' or 'test'")


    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        gt_img = Image.open(self.gt_list[idx]).convert('RGB')

        if self.lq_list[idx]:
            lq_img = Image.open(self.lq_list[idx]).convert('RGB')
        else:
            raise ValueError('LQ dataset list is not provided.')

        gt_img = img2tensor([np.asarray(gt_img)/255.], bgr2rgb=False, float32=True)[0].unsqueeze(0).to('cpu')
        lq_img = img2tensor([np.asarray(lq_img)/255.], bgr2rgb=False, float32=True)[0].unsqueeze(0).to('cpu')

        # images scaled to -1,1
        gt_img = F.normalize(gt_img.squeeze(0), mean=[0.5], std=[0.5])
        lq_img = F.normalize(lq_img.squeeze(0), mean=[0.5], std=[0.5])
        # print("=================", lq_img.shape)
        example = {}
        example["neg_prompt"] = self.args.neg_prompt
        example["null_prompt"] = ""
        example["output_pixel_values"] = gt_img  # shape: 3, h, w
        example["conditioning_pixel_values"] = lq_img
        example["image_gt_name"] = os.path.basename(self.gt_list[idx])
        # print(f'{self.split}=====> Loaded image pair {self.gt_list[idx]} and {self.lq_list[idx]}')
        return example
        