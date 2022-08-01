import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.unet_modify import Unet


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(input)
        loss = loss_type(output, target, maximum)
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
                f'Model parameters = {model.parameters()}',
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


def load_pretrained_weights(model, pretrained_copy):
    model.state_dict()['first_block.layers.0.weight'] = pretrained_copy['down_sample_layers.0.layers.0.weight']
    model.state_dict()['first_block.layers.3.weight'] = pretrained_copy['down_sample_layers.0.layers.4.weight']
    model.state_dict()['down1.layers.2.layers.0.weight'] = pretrained_copy['down_sample_layers.1.layers.0.weight']
    model.state_dict()['down1.layers.2.layers.3.weight'] = pretrained_copy['down_sample_layers.1.layers.4.weight']
    model.state_dict()['down2.layers.2.layers.0.weight'] = pretrained_copy['down_sample_layers.2.layers.0.weight']
    model.state_dict()['down2.layers.2.layers.3.weight'] = pretrained_copy['down_sample_layers.2.layers.4.weight']
    model.state_dict()['down3.layers.2.layers.0.weight'] = pretrained_copy['down_sample_layers.3.layers.0.weight']
    model.state_dict()['down3.layers.2.layers.3.weight'] = pretrained_copy['down_sample_layers.3.layers.4.weight']
    model.state_dict()['down4.layers.2.layers.0.weight'] = pretrained_copy['conv.layers.0.weight']
    model.state_dict()['down4.layers.2.layers.3.weight'] = pretrained_copy['conv.layers.4.weight']

    model.state_dict()['up1.conv.layers.0.weight'] = pretrained_copy['up_conv.0.layers.0.weight']
    model.state_dict()['up1.conv.layers.3.weight'] = pretrained_copy['up_conv.0.layers.4.weight']
    model.state_dict()['up2.conv.layers.0.weight'] = pretrained_copy['up_conv.1.layers.0.weight']
    model.state_dict()['up2.conv.layers.3.weight'] = pretrained_copy['up_conv.1.layers.4.weight']
    model.state_dict()['up3.conv.layers.0.weight'] = pretrained_copy['up_conv.2.layers.0.weight']
    model.state_dict()['up3.conv.layers.3.weight'] = pretrained_copy['up_conv.2.layers.4.weight']
    model.state_dict()['up4.conv.layers.0.weight'] = pretrained_copy['up_conv.3.0.layers.0.weight']
    model.state_dict()['up4.conv.layers.3.weight'] = pretrained_copy['up_conv.3.0.layers.4.weight']
    model.state_dict()['last_block.weight'] = pretrained_copy['up_conv.3.1.weight']
    model.state_dict()['last_block.bias'] = pretrained_copy['up_conv.3.1.bias']
    return model


def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    pretrained = torch.load(
        '/content/drive/Shareddrives/2022 FastMRI/pretrained_models/brain_leaderboard_state_dict_unet.pt')
    pretrained_copy = copy.deepcopy(pretrained)

    model = Unet(in_chans=args.in_chans, out_chans=args.out_chans)
    model.to(device=device)

    model = load_pretrained_weights(model, pretrained_copy)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

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