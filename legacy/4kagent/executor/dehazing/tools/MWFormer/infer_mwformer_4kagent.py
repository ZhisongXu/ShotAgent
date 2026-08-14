import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData, ValData_unpaired_demo
from utils_val import validation_stylevec, validation_unpaired
import numpy as np
import random
from model.EncDec import Network_top    #default
from model.style_filter64 import StyleFilter_Top


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
# Basic settings
parser.add_argument('--val_batch_size', type=int, default=1,
                    help='Validation/test batch size')
parser.add_argument('--seed', type=int, default=19,
                    help='Random seed for reproducibility')
# Model checkpoint paths
parser.add_argument('--restore-from-stylefilter', type=str,
                    default='./checkpoints/MWFormer_L/style_filter',
                    help='Path to the weights of the style filter network')
parser.add_argument('--restore-from-backbone', type=str,
                    default='./checkpoints/MWFormer_L/backbone',
                    help='Path to the weights of the image restoration backbone')
# Validation data and output
parser.add_argument('--val_data_dir', type=str,
                    default='./input',
                    help='Directory containing validation data')
parser.add_argument('--val_filename', type=str,
                    default='raindroptesta.txt',
                    help='Filename listing the validation data')
parser.add_argument('--result_dir', type=str,
                    default='./output',
                    help='Directory to save the validation results')
args = parser.parse_args()

val_batch_size = args.val_batch_size
result_dir = args.result_dir

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = args.val_data_dir
val_filename = args.val_filename ## This text file should contain all the names of the images and must be placed in val data directory

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #
# val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)
val_data_loader = DataLoader(ValData_unpaired_demo(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)


# --- the network backbone --- #
net = Network_top().cuda()
net = nn.DataParallel(net, device_ids=device_ids)
weights_dict = torch.load(args.restore_from_backbone)
net.load_state_dict(weights_dict)
net.eval()

# --- the style filter --- #
StyleFilter = StyleFilter_Top() 
StyleFilter.to(device)
StyleFilter = nn.DataParallel(StyleFilter, device_ids=device_ids)
weights_dict = torch.load(args.restore_from_stylefilter)
StyleFilter.load_state_dict(weights_dict)
for param in StyleFilter.parameters():
    param.require_grad = False
StyleFilter.eval()


# --- Use the evaluation model in testing --- #
print('--- Testing starts! ---')
start_time = time.time()
with torch.no_grad():
    # val_psnr, val_ssim = validation_stylevec(StyleFilter, net, val_data_loader, device)
    validation_unpaired(StyleFilter, net, val_data_loader, device, result_dir)
    
end_time = time.time() - start_time
print('validation time is {0:.4f}'.format(end_time))


