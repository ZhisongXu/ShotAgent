# modified from https://github.com/IDKiro/DehazeFormer/blob/main/test.py

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader, SingleLoader
from models import *


parser = argparse.ArgumentParser()
# parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--model', default='dehazeformer-b', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
# parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
# parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')

parser.add_argument('--tile_size', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	# PSNR = AverageMeter()
	# SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	# os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	os.makedirs(os.path.join(result_dir), exist_ok=True)
	# f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		# input = batch['source'].cuda()
		# target = batch['target'].cuda()
		input = batch['img'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			# output = network(input).clamp_(-1, 1)

			# # [-1, 1] to [0, 1]
			# output = output * 0.5 + 0.5
			# # target = target * 0.5 + 0.5

			# # psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			# # _, _, H, W = output.size()
			# # down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			# # ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
			# # 				F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
			# # 				data_range=1, size_average=False).item()		
			b, c, h, w = input.shape
			if args.tile_size is None or max(h, w) <= args.tile_size:
				restored = network(input)
				# restored = restored[0]
			else:
				# test the image tile by tile
				tile = min(args.tile_size, h, w)
				tile = tile - (tile % 8)
				assert tile % 8 == 0, "tile size should be multiple of 8"
				tile_overlap = args.tile_overlap

				stride = tile - tile_overlap
				h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
				w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
				E = torch.zeros(b, c, h, w).type_as(input)
				W = torch.zeros_like(E)

				for h_idx in h_idx_list:
					for w_idx in w_idx_list:
						in_patch = input[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
						out_patch = network(in_patch)[0]
						out_patch_mask = torch.ones_like(out_patch)

						E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
						W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
				restored = E.div_(W)
				# print("restored shape: ", restored.shape)		

			output = restored.clamp_(-1, 1)
			output = output * 0.5 + 0.5
		# PSNR.update(psnr_val)
		# SSIM.update(ssim_val)

		# print('Test: [{0}]\t'
		# 	  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
		# 	  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
		# 	  .format(idx, psnr=PSNR, ssim=SSIM))

		# f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		# out_img = chw_to_hwc(output.detach().cpu().numpy())
		# write_img(os.path.join(result_dir, 'imgs', filename), out_img)
		write_img(os.path.join(result_dir, filename), out_img)

	# return out_img
	# f_result.close()

	# os.rename(os.path.join(result_dir, 'results.csv'), 
			#   os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.model+'.pth')
	print(saved_model_dir)

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	# test_dataset = PairLoader(dataset_dir, 'test', 'test')
	dataset_dir = os.path.join(args.data_dir)
	test_dataset = SingleLoader(dataset_dir)
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	# result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	# result_dir = os.path.join(args.result_dir)
	result_dir = args.result_dir
	test(test_loader, network, result_dir)