import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import copy

from collections import defaultdict
from unet.utils.data.load_data_two_channel import create_data_loaders
from unet.utils.common.utils import save_reconstructions, ssim_loss
from unet.utils.common.loss_function import SSIMLoss
from unet.utils.model.unet_attn_two_channel import Att_UNet
from torch.optim import lr_scheduler

def mixup(input, target, maximum, alpha = 0.4):
    batch_size = input.shape[0]
    l = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size).cuda()

    mixed_input = l * input + (1 - l) * input[idx]
    mixed_target = l * target + (1 - l) * target[idx]
    mixed_max = torch.tensor([mixed_target[i].max() for i in range(batch_size)])

    return mixed_input, mixed_target, mixed_max

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, target, maximum, _, _ = data
        input, target, maximum = mixup(input, target, maximum)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(input)
        # print(input)
        # print(output)
        # print(target)
        # print("===================")
        loss = loss_type(output, target, maximum)
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output = model(input)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, val_loss):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.tar'
    )
    # if is_new_best:
    #     shutil.copyfile(exp_dir / 'model.tar', exp_dir / 'best_model.tar')


def save_best_model(args, exp_dir, epoch, model, optimizer, best_val_loss):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'best_model.tar'
    )
def load_model(exp_dir):
    checkpoint = torch.load(exp_dir / 'model.tar', map_location=torch.device("cpu"))
    return checkpoint

def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = Att_UNet(num_class=args.out_chans)
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 25], gamma=0.5)

    ckp = load_model(args.exp_dir)
    best_val_loss = ckp['val_loss'].item()
    print(best_val_loss)
    start_epoch = ckp['epoch']
    print(start_epoch)
    model.load_state_dict(ckp['model'])

    # best_val_loss = 1.
    # start_epoch = 0

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        scheduler.step()

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, val_loss)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
            save_best_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss)
