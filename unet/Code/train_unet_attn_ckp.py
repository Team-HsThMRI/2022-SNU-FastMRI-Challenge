import argparse
import shutil
from unet.utils.learning.train_part_unet_attn_ckp import train
from unet.utils.model.random_seed import seed_fix
from pathlib import Path
import torch

def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=200, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_Unet_attn', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/input/train/image/',
                        help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/input/val/image/',
                        help='Directory of validation data')

    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    # dictionary 형태로 저장되는 h5 파일의 key 이름을 설정하는 부분
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    args = parser.parse_args(args=[])
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '/content/drive/MyDrive/FastMRI_LHS/result' / args.net_name / 'checkpoints'
    args.val_dir = '/content/drive/MyDrive/FastMRI_LHS/result' / args.net_name / 'reconstructions_val'

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    seed_fix(42)

    train(args)
