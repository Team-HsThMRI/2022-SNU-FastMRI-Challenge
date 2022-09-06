import argparse
import shutil
from utils.learning.train_part_varnet_with_attn_unet import train
from utils.model.random_seed import seed_fix
from pathlib import Path
import torch

import sys
sys.path.append('/root/fastMRI/utils/model')

# if os.getcwd() + '/utils/model/' not in sys.path:
#     sys.path.insert(1, os.getcwd() + '/utils/model/')

def parse():

    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet_with_attn_Unet', help='Name of network')
    parser.add_argument('-kt', '--data-path-train-kspace', type=Path, default='/root/input/val/',
                        help='Directory of train kspace data')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/input/train/image/',
                        help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/fastMRI/model/test_varnet_with_attn_Unet/intermediate_varnet_val/reconstructions_val/',
                        help='Directory of validation data')

    parser.add_argument('--cascade', type=int, default=6, help='Number of cascades | Should be less than 12')  ## important hyperparameter
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    # dictionary 형태로 저장되는 h5 파일의 key 이름을 설정하는 부분
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--input-key-kspace', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    args = parser.parse_args(args=[])
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../model' / args.net_name / 'checkpoints'
    args.val_dir_1 = '../model' / args.net_name / 'intermediate_varnet/reconstructions_val'
    args.val_dir_2 = '../model' / args.net_name / 'reconstructions_val'
    args.main_dir = '../model' / args.net_name / __file__

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir_1.mkdir(parents=True, exist_ok=True)
    args.val_dir_2.mkdir(parents=True, exist_ok=True)

    seed_fix(42)

    train(args)
