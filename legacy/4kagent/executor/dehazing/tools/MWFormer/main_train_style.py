#Train a style filter to generate a 64-dimension vector to represent the weather type.
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F 
from perceptual import LossNetwork
from torchvision.models import vgg16

import numpy as np
import random
from tqdm import tqdm
from datetime import datetime

from model.style_filter64 import StyleFilter_Top     #style discriminator
from train_data_functions import TrainingDataset, validation_train
from configs.train_style_config import get_arguments
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM_stylefilter = 'without_pretraining'


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
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'

    cudnn.enabled = True
    # --- Gpu device --- #
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #instantiate style discriminators, and their optimizers
    StyleFilter = StyleFilter_Top()  #input size: c*(c+1)/2   
    StyleFilter_optimizer = torch.optim.Adamax([p for p in StyleFilter.parameters() if p.requires_grad == True], lr=args.lr_style)
    StyleFilter.to(device)
    StyleFilter = nn.DataParallel(StyleFilter, device_ids=device_ids)
    for param in StyleFilter.parameters():
        param.requires_grad = True

    if args.restore_from_stylefilter != RESTORE_FROM_stylefilter:
        #restore = torch.load(args.restore_from_stylefilter)
        #StyleFilter.load_state_dict(restore)
        restore = torch.load(args.restore_from_stylefilter, map_location=lambda storage, loc: storage).module.state_dict()
        weights_dict = {}
        for k, v in restore.items():
            new_k = 'module.' + k
            weights_dict[new_k] = v
        StyleFilter.load_state_dict(weights_dict)
        
        


    #contrastive loss for style discriminator
    stylefilter_loss = losses.ContrastiveLoss(
        pos_margin=0.5,     
        neg_margin=0,     
        distance=CosineSimilarity(),
        reducer=MeanReducer()
        )
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    crop_size = args.crop_size
    train_data_dir = args.train_data_dir
    labeled_name = args.labeled_name

    raindrop_dataset = TrainingDataset(0, crop_size, train_data_dir,labeled_name)  #load raindrop dataset (drop last)
    raindrop_loader_style = data.DataLoader(raindrop_dataset, batch_size=args.batch_size_style, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=True)
    raindrop_loader_style_iter = enumerate(raindrop_loader_style)

    rain_dataset = TrainingDataset(1, crop_size, train_data_dir,labeled_name)  #load rain dataset
    rain_loader_style = data.DataLoader(rain_dataset,batch_size=args.batch_size_style, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=True)
    rain_loader_style_iter = enumerate(rain_loader_style)

    snow_dataset = TrainingDataset(2, crop_size, train_data_dir,labeled_name)  #load snow dataset
    snow_loader_style = data.DataLoader(snow_dataset,batch_size=args.batch_size_style, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=True)
    snow_loader_style_iter = enumerate(snow_loader_style)

    # define arrays
    style_loss_arr = np.array([])
    style_loss_pth = args.snapshot_dir + '/style_loss.npy'
    np.save(file=style_loss_pth, arr=style_loss_arr)
    total_style_loss = 0        # save at a certain interval
    cnt = 0

    # start training
    print('Start training...')
    save_pred_every = args.save_pred_every
    print('loss_save_step: ', args.loss_save_step)
    for i_iter in tqdm(range(0, args.num_steps)): 
        if i_iter == args.num_steps / 2:
            lr = args.lr_style / 2
            for param_group in StyleFilter_optimizer.param_groups:
                param_group['lr'] = lr

        cnt = cnt + 1
        StyleFilter_optimizer.zero_grad()
        # train style filtering module
  
        try:    #raindrop
            _, batch = raindrop_loader_style_iter.__next__()
        except StopIteration:
            raindrop_loader_style = data.DataLoader(raindrop_dataset, batch_size=args.batch_size_style, shuffle=True, num_workers=args.num_workers,
                                    pin_memory=True, drop_last=True)    #load raindrop dataset (drop last)
            raindrop_loader_style_iter = enumerate(raindrop_loader_style) 
            _, batch = raindrop_loader_style_iter.__next__()
        raindrop_img, gt_raindrop_img = batch
            
        try:    #rain
            _, batch = rain_loader_style_iter.__next__()
        except StopIteration:
            rain_loader_style = data.DataLoader(rain_dataset, batch_size=args.batch_size_style, shuffle=True, num_workers=args.num_workers,
                                    pin_memory=True, drop_last=True)    #load raindrop dataset (drop last)
            rain_loader_style_iter = enumerate(rain_loader_style) 
            _, batch = rain_loader_style_iter.__next__()
        rain_img, gt_rain_img = batch

        try:    #snow
            _, batch = snow_loader_style_iter.__next__()
        except StopIteration:
            snow_loader_style = data.DataLoader(snow_dataset, batch_size=args.batch_size_style, shuffle=True, num_workers=args.num_workers,
                                    pin_memory=True, drop_last=True)    #load raindrop dataset (drop last)
            snow_loader_style_iter = enumerate(snow_loader_style) 
            _, batch = snow_loader_style_iter.__next__()
        snow_img, gt_snow_img = batch

        StyleFilter.train()  
        StyleFilter_optimizer.zero_grad()

        raindrop_vec = StyleFilter(raindrop_img)
        rain_vec = StyleFilter(rain_img)
        snow_vec = StyleFilter(snow_img)
        gt_vec = StyleFilter(torch.cat([gt_raindrop_img, gt_rain_img, gt_snow_img], dim=0))  #3*batch_size_style
        #print('vec shape: ', snow_vec.shape)                                                                                                                                                                                      

        style_embeddings = torch.cat((torch.unsqueeze(raindrop_vec[0],0),torch.unsqueeze(rain_vec[0],0),torch.unsqueeze(snow_vec[0],0),
                                      torch.unsqueeze(raindrop_vec[1],0),torch.unsqueeze(rain_vec[1],0),torch.unsqueeze(snow_vec[1],0),
                                      torch.unsqueeze(raindrop_vec[2],0),torch.unsqueeze(rain_vec[2],0),torch.unsqueeze(snow_vec[2],0),
                                      torch.unsqueeze(raindrop_vec[3],0),torch.unsqueeze(rain_vec[3],0),torch.unsqueeze(snow_vec[3],0),
                                      torch.unsqueeze(raindrop_vec[4],0),torch.unsqueeze(rain_vec[4],0),torch.unsqueeze(snow_vec[4],0),
                                      torch.unsqueeze(raindrop_vec[5],0),torch.unsqueeze(rain_vec[5],0),torch.unsqueeze(snow_vec[5],0),
                                      torch.unsqueeze(raindrop_vec[6],0),torch.unsqueeze(rain_vec[6],0),torch.unsqueeze(snow_vec[6],0),
                                      torch.unsqueeze(raindrop_vec[7],0),torch.unsqueeze(rain_vec[7],0),torch.unsqueeze(snow_vec[7],0),
                                      torch.unsqueeze(gt_vec[0],0),      torch.unsqueeze(gt_vec[3],0),  torch.unsqueeze(gt_vec[6],0),
                                      torch.unsqueeze(gt_vec[9],0),      torch.unsqueeze(gt_vec[12],0), torch.unsqueeze(gt_vec[15],0),
                                      torch.unsqueeze(gt_vec[18],0),     torch.unsqueeze(gt_vec[21],0)),0)       # batch size = 8 ?
        #print('style embedding shape: ', style_embeddings.shape)

        style_labels = torch.LongTensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 
                                         0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                                         3, 3, 3, 3, 3, 3, 3, 3])  # style labels of style embeddings
        style_filter_loss = stylefilter_loss(style_embeddings,style_labels)    # contrastive loss
        total_style_loss += style_filter_loss.item()

        if i_iter % args.loss_save_step == 0:
            total_style_loss = total_style_loss / cnt
            style_loss_arr = np.load(file=style_loss_pth)
            style_loss_arr = np.append(style_loss_arr, total_style_loss)
            np.save(file=style_loss_pth, arr=style_loss_arr)
            print('iter: ', i_iter)
            print('total style loss: ', total_style_loss)

            total_style_loss = 0
            cnt = 0

        style_filter_loss.backward(retain_graph=False)
        StyleFilter_optimizer.step()     # update the parameters of the style discriminator
      
        if i_iter == args.num_steps-1:      #save style discriminators in the end 
            torch.save(StyleFilter.state_dict(), osp.join(args.snapshot_dir, run_name)+'_stylefilter_'+str(i_iter)+'.pth')

        if i_iter % save_pred_every == 0 and i_iter != 0:       #checkpoint
            print('taking snapshot ...')
            torch.save(StyleFilter.state_dict(), osp.join(args.snapshot_dir, run_name)+'_stylefilter_'+str(i_iter)+'.pth')
            
if __name__ == '__main__':
    main()