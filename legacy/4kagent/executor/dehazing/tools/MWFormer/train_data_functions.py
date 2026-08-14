import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class TrainingDataset(data.Dataset):
    # type 0: raindrop;  type 1: rain;  type 2: snow;  type 3: full
    def __init__(self, type, crop_size, train_data_dir, train_filename):
        super().__init__()
        train_list = train_data_dir + train_filename    #txt filename
        with open(train_list) as f:
            contents = f.readlines()
            if type==0:
                input_names = [i[13:].strip() for i in contents if i.strip().split("/")[-1].endswith("rain.png")]  
            elif type==1:
                input_names = [i[13:].strip() for i in contents if i.split("/")[-1].startswith("im")]
            elif type==2:
                input_names = [i[13:].strip() for i in contents if i.strip().split("/")[-1].endswith("jpg")]
            else:
                input_names = [i[13:].strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

        print("length of input names: ", len(input_names))
        
    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        #img_id = re.split('/',input_name)[-1][:-4]

        input_img = Image.open(self.train_data_dir + input_name)

        try:
            gt_img = Image.open(self.train_data_dir + gt_name)
        except:
            gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(input_im.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return input_im, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
    
#snow: end with "jpg"
#raindrop: end with "rain.png"
#rain: start with "im", end with "png"


#cal psnr
def calc_psnr(im1, im2):  #PSNR of Y channel

    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()


    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_psnr(im1_y, im2_y)]
    return ans

def calc_ssim(im1, im2):
    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]

def validation_train(net, val_data_loader, device):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            _,_,pred_image = net(input_im)
            pred_image = pred_image.clamp(0,1)

# --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(gt, pred_image))

        # --- Calculate the average SSIM --- #
        #ssim_list.extend(calc_ssim(pred_image, gt))

    avr_psnr = sum(psnr_list) / len(psnr_list)
    #avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr