#local + global + channel

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F 
import itertools
from perceptual import LossNetwork
from torchvision.models import vgg16

import numpy as np
import random
from tqdm import tqdm
from datetime import datetime

from model.EncDec import Network_top      #image restoration network
from model.style_filter64 import StyleFilter_Top     #style discriminator
from train_data_functions import TrainingDataset
from utils_network import validation_train
from configs.train_main_config import get_arguments

RESTORE_FROM = 'without_pretraining'

def loss_calc(pred, gt, loss_network, lambda_loss):
    smooth_loss = F.smooth_l1_loss(pred, gt)
    perceptual_loss = loss_network(pred, gt)
    return smooth_loss + lambda_loss*perceptual_loss


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

def main():
    """Create the model and start the training."""

    args = get_arguments()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'

    cudnn.enabled = True
    # --- Gpu device --- #
    device_ids = [1,0]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #instantiate the encoder-decoder
    if args.restore_from == RESTORE_FROM:  #default: without pretraining
        start_iter = 0
        model = Network_top().to(device)
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        restore = torch.load(args.restore_from, map_location=lambda storage, loc: storage).module.state_dict()
        weights_dict = {}
        for k, v in restore.items():
            new_k = 'module.' + k
            weights_dict[new_k] = v
        model = Network_top().to(device)
        model = nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(weights_dict)
        #restore = torch.load(args.restore_from, map_location=lambda storage, loc: storage)["state_dict"]
        #model = Network_top().to(device)
        #model = nn.DataParallel(model, device_ids=device_ids)
        #model.load_state_dict(restore)
        start_iter = 0

    model.train()

    #instantiate style discriminators, and their optimizers
    StyleFilter = StyleFilter_Top()  #input size: c*(c+1)/2   #2080?
    StyleFilter.to(device)
    StyleFilter = nn.DataParallel(StyleFilter, device_ids=device_ids)
    restore = torch.load(args.restore_from_stylefilter)
    StyleFilter.load_state_dict(restore)
    for param in StyleFilter.parameters():      # if don't train the StyleFilter
        param.require_grad = False
    StyleFilter.eval()
    #StyleFilter.train()

    lambda_loss = args.lambda_loss #loss weight
    
    #initialize perceptual loss model
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    crop_size = args.crop_size
    train_data_dir = args.train_data_dir
    labeled_name = args.labeled_name

    raindrop_dataset = TrainingDataset(0, crop_size, train_data_dir,labeled_name)
    raindrop_loader = data.DataLoader(raindrop_dataset, batch_size=10, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=True)    #load raindrop dataset (drop last)
    raindrop_loader_iter = enumerate(raindrop_loader)

    rain_dataset = TrainingDataset(1, crop_size, train_data_dir,labeled_name)
    rain_loader = data.DataLoader(rain_dataset,batch_size=11, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=True)    #load rain dataset
    rain_loader_iter = enumerate(rain_loader)


    snow_dataset = TrainingDataset(2, crop_size, train_data_dir,labeled_name)
    snow_loader = data.DataLoader(snow_dataset,batch_size=11, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=True)    #load snow dataset
    snow_loader_iter = enumerate(snow_loader)

    full_dataset = TrainingDataset(3, crop_size, train_data_dir,labeled_name)
    train_size = int(0.95 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    val_data_loader = data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=args.num_workers, pin_memory=True)      # batch size?

    main_lr = args.learning_rate
    opts = torch.optim.Adam(model.parameters(), lr=args.learning_rate)    #optimizer for only the main network

    # define arrays
    val_psnr_arr = np.array([])
    val_psnr_pth = args.snapshot_dir + '/val_psnr.npy'
    #if not(os.path.exists('./checkpoint')):
    #    os.mkdir('./checkpoint')
    np.save(file=val_psnr_pth, arr=val_psnr_arr)
    print("testing...")
    old_val_psnr, old_val_ssim = validation_train(StyleFilter, model, val_data_loader, device)
    print('initial model PSNR: ', old_val_psnr)

    processing_loss_arr = np.array([])
    processing_loss_pth = args.snapshot_dir + '/processing_loss.npy'
    np.save(file=processing_loss_pth, arr=processing_loss_arr)

    # start training
    print('Start training...')
    for i_iter in tqdm(range(start_iter, args.num_steps)): 

        if i_iter==50000 or i_iter==75000:
            main_lr = main_lr / 2
            for param_group in opts.param_groups:
                param_group['lr'] = main_lr

        opts.zero_grad()
            
        # train image processing network using smooth l1 loss and perceptual loss
        # freeze the parameters of style filtering modules
        try:    #raindrop
            _, batch_raindrop = raindrop_loader_iter.__next__()
        except StopIteration:
            raindrop_loader = data.DataLoader(raindrop_dataset, batch_size=10, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)    #load raindrop dataset (drop last)
            raindrop_loader_iter = enumerate(raindrop_loader) 
            _, batch_raindrop = raindrop_loader_iter.__next__()
        raindrop_img, gt_raindrop_img = batch_raindrop
        imgs_raindrop = Variable(raindrop_img).to(device)
        try:    #rain
            _, batch_rain = rain_loader_iter.__next__()
        except StopIteration:
            rain_loader = data.DataLoader(rain_dataset, batch_size=11, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)    #load raindrop dataset (drop last)
            rain_loader_iter = enumerate(rain_loader) 
            _, batch_rain = rain_loader_iter.__next__()
        rain_img, gt_rain_img = batch_rain
        imgs_rain = Variable(rain_img).to(device)
        try:    #snow
                _, batch_snow = snow_loader_iter.__next__()
        except StopIteration:
            snow_loader = data.DataLoader(snow_dataset, batch_size=11, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)    #load raindrop dataset (drop last)
            snow_loader_iter = enumerate(snow_loader) 
            _, batch_snow = snow_loader_iter.__next__()
        snow_img, gt_snow_img = batch_snow
        imgs_snow = Variable(snow_img).to(device)

        imgs_input = torch.cat([imgs_raindrop, imgs_rain, imgs_snow], dim=0)
        gt = torch.cat([gt_raindrop_img, gt_rain_img, gt_snow_img], dim=0)
        feature_vec = StyleFilter(imgs_input)
        pred = model(imgs_input, feature_vec)
        loss_p = loss_calc(pred, gt.to(device), loss_network, lambda_loss)

        # image processing loss
        loss = loss_p
        loss.backward()
        loss_p_value = loss_p.data.cpu().numpy()       # image processing loss

        if i_iter % args.loss_save_step == 0:
            processing_loss_arr = np.load(file=processing_loss_pth)
            processing_loss_arr = np.append(processing_loss_arr, loss_p_value)
            np.save(file=processing_loss_pth, arr=processing_loss_arr)
            print('image processing loss:', loss_p_value)

        opts.step()        # update the parameters of the image processing network

        if i_iter >= args.num_steps_stop - 1:  #save model in the end
            print('save model ..')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.file_name + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:       #checkpoint
            print('taking snapshot ...')
            torch.save({
                'state_dict':model.state_dict(),
                'train_iter':i_iter,
                'args':args
            },osp.join(args.snapshot_dir, run_name)+'_backbone'+str(i_iter)+'.pth')

        if i_iter % args.loss_save_step == 0 and i_iter != 0:      #save best
            print('testing...')
            val_psnr, val_ssim = validation_train(StyleFilter, model, val_data_loader, device)
            val_psnr_arr = np.load(file=val_psnr_pth)
            val_psnr_arr = np.append(val_psnr_arr, val_psnr)
            np.save(file=val_psnr_pth, arr=val_psnr_arr)
            print('iter: ', i_iter)
            print('psnr: ', val_psnr)
            print('ssim: ', val_ssim)
            if val_psnr >= old_val_psnr:
                torch.save(model, args.snapshot_dir + '/best_all')
                print('model saved')
                old_val_psnr = val_psnr
            
if __name__ == '__main__':
    main()