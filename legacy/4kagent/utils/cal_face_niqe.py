import argparse
import cv2
import glob
import math
import numpy as np
import os
import torch
import pyiqa


def img2tensor(img, bgr2rgb, float32):
    '''
    Convert a numpy array image to a torch tensor.

    Args:
        img (ndarray): Input image.
        bgr2rgb (bool): Whether to change BGR to RGB.
        float32 (bool): Whether to convert to float32.
    '''
    if img.shape[2] == 3 and bgr2rgb:
        if img.dtype == 'float64':
            img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    if float32:
        img = img.float()
    return img


def calculate_niqe(restored_path):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    iqa_niqe = pyiqa.create_metric('niqe').to(device)

    img = cv2.imread(restored_path)
    img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).contiguous().to(device)
    niqe = iqa_niqe(img).item()
    print(f'{restored_path} - NIQE : {niqe}')
        
    
    return niqe

