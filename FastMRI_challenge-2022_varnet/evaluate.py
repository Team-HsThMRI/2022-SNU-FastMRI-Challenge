import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part import forward

def parse():
    parser = argparse.ArgumentParser(description='Test Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net-name', type=Path, default='test_for_leaderboard', help='Name of network')
    parser.add_argument('-p', '--data-path-kspace', type=Path, default='/root/input/leaderboard', help='Directory of test kspace data')
    parser.add_argument('-i', '--data-path-image', type=Path, default='/root/input/leaderboard/image', help='Directory of test image data')
    
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--cascade', type=int, default=6,
                        help='Number of cascades | Should be less than 12')  ## important hyperparameter

    parser.add_argument("--input-key", type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--input-key-image', type=str, default='image_input', help='Name of input key')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '/root/fastMRI/result' / args.net_name / 'checkpoints'
    args.forward_dir = '/root/fastMRI/result' / args.net_name / 'reconstructions_forward'
    args.main_dir = '../result' / args.net_name / __file__
    print(args.forward_dir)
    forward(args)

