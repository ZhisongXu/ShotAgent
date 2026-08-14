import argparse
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
BETA = 0.005
BATCH_SIZE = 32
BATCH_SIZE_STYLE = 8
ITER_SIZE = 1
NUM_WORKERS = 4
NUM_CLASSES = 19 
NUM_STEPS = 10000  
NUM_STEPS_STOP = NUM_STEPS  
RANDOM_SEED = 1234
RESTORE_FROM = 'without_pretraining'
RESTORE_FROM_stylefilter = 'without_pretraining'
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = f'./checkpoints/exp1/'
TRAIN_DATA_DIR = './data/train/allweather/'
LABELED_NAME = 'allweather.txt'

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-batch-size-style", type=int, default=BATCH_SIZE_STYLE)
    parser.add_argument("-num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("-num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("-num-steps-stop", type=int, default=NUM_STEPS_STOP)
    parser.add_argument("-random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("-restore-from-stylefilter", type=str, default=RESTORE_FROM_stylefilter)
    parser.add_argument("-save-pred-every", type=int, default=SAVE_PRED_EVERY)  #checkpoint interval
    parser.add_argument("-snapshot-dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("-file-name", type=str, required=True)
    parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
    parser.add_argument('-lr_style', help='Set the learning rate', default=2e-4, type=float)       #2e-4
    parser.add_argument('-loss_save_step', default=100, type=int)
    parser.add_argument('-train_data_dir', type=str, default=TRAIN_DATA_DIR)
    parser.add_argument('-labeled_name', type=str, default=LABELED_NAME)
    return parser.parse_args()

args = get_arguments()