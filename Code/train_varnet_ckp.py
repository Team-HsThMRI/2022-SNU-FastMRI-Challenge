import torch
import argparse
import shutil
import os, sys

import sys
sys.path.append('/root/fastMRI/utils/model')

from utils.learning.train_part_ckp import train
from pathlib import Path
from utils.model.random_seed import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    # parser.add_argument('-t', '--data-path-train', type=Path, default='/content/drive/Shareddrives/2022 FastMRI/train_real/', help='Directory of train data')
    # parser.add_argument('-v', '--data-path-val', type=Path, default='/content/drive/Shareddrives/2022 FastMRI/val_real/', help='Directory of validation data')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/content/drive/MyDrive/train_real/',
                        help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/content/drive/MyDrive/val_real/',
                        help='Directory of validation data')

    parser.add_argument('--cascade', type=int, default=6,
                        help='Number of cascades | Should be less than 12')  ## important hyperparameter
    parser.add_argument('--input-key-kspace', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    parser.add_argument('-f')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../model' / args.net_name / 'checkpoints'
    args.val_dir = '../model' / args.net_name / 'reconstructions_val'
    # args.main_dir = '../model' / args.net_name / __file__

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    seed_fix(42)
    train(args)