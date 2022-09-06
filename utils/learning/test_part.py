import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.load_data_leaderboard_unet import create_data_loaders_image
from utils.data.load_data_two_channel import create_data_loaders as create_data_loaders_image_2chans
from utils.data.load_data_one_channel import create_data_loaders as create_data_loaders_image_1chan

from utils.model.dircn.dircn_unet import DIRCN as DIRCN_unet
from utils.model.dircn.dircn_attn import DIRCN as DIRCN_attn
from utils.model.unet_attn_two_channel import Att_UNet
from utils.model.unet_attn_one_chan import Att_UNet as Att_UNet_one
from utils.model.varnet_ckp import VarNet


def test_kspace(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            print(fnames)
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)
            print(output.shape)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        print(f"{fname} recon")
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def test_image(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output = model(input)
            print(output.shape)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device :', torch.cuda.current_device())

    checkpoint1 = torch.load('/root/fastMRI/model/test_dircn/checkpoints/best_model_dircn_with_unet.pt', map_location=torch.device("cpu"))
    checkpoint2 = torch.load('/root/fastMRI/model/test_dircn_attn/checkpoints/best_model_dircn_with_attnunet.pt', map_location=torch.device("cpu"))
    checkpoint3 = torch.load('/root/fastMRI/model/test_Unet_attn/checkpoints/best_model_attnunet.tar', map_location=torch.device("cpu"))
    checkpoint4 = torch.load('/root/fastMRI/model/test_varnet/checkpoints/best_model_varnet.pt', map_location=torch.device("cpu"))

    model1 = DIRCN_unet(num_cascades=args.cascade1)
    model1.to(device=device)
    print(checkpoint1['epoch'], checkpoint1['best_val_loss'].item())
    model1.load_state_dict(checkpoint1['model'])

    model2 = DIRCN_attn(num_cascades=args.cascade2)
    model2.to(device=device)
    print(checkpoint2['epoch'], checkpoint2['best_val_loss'].item())
    model2.load_state_dict(checkpoint2['model'])

    model3 = Att_UNet(num_class=2)
    model3.to(device=device)
    print(checkpoint3['epoch'], checkpoint3['best_val_loss'].item())
    model3.load_state_dict(checkpoint3['model'])

    model4 = VarNet(states=checkpoint4['model'], num_cascades=args.cascade1)
    model4.to(device=device)
    print(checkpoint4['epoch'], checkpoint4['best_val_loss'].item())
    model4.load_state_dict(checkpoint4['model'])

    forward_loader_kspace = create_data_loaders(data_path=args.data_path_kspace, args=args, isforward=True)
    # torch.save(forward_loader_kspace, args.exp_dir / 'kspace_loader.pt')
    # forward_loader_kspace = torch.load(args.exp_dir / 'kspace_loader.pt')
    print("*************")
    print(len(forward_loader_kspace))

    forward_loader_image = create_data_loaders_image(data_path=args.data_path_image, args=args, isforward=True)
    # torch.save(forward_loader_image, args.exp_dir / 'image_loader.pt')
    # forward_loader_image = torch.load(args.exp_dir / 'image_loader.pt')

    forward_loader_image_2chans = create_data_loaders_image_2chans(data_path=args.data_path_image, args=args, isforward=True)
    # torch.save(forward_loader_image_2chans, args.exp_dir / 'image_loader_2chans.pt')
    # forward_loader_image_2chans = torch.load(args.exp_dir / 'image_loader_2chans.pt')


    reconstructions_1, inputs = test_kspace(args, model1, forward_loader_kspace)
    reconstructions_2, inputs = test_kspace(args, model2, forward_loader_kspace)
    reconstructions_3, inputs = test_image(args, model3, forward_loader_image_2chans)
    reconstructions_4, inputs = test_kspace(args, model4, forward_loader_kspace)

    for fname, _ in reconstructions_1.items():
        reconstructions_1[fname] = reconstructions_1[fname] * 0.25 + reconstructions_2[fname] * 0.25 + reconstructions_3[fname] * 0.25 + reconstructions_4[fname] * 0.25

    save_reconstructions(reconstructions_1, args.forward_dir, inputs=inputs)
    