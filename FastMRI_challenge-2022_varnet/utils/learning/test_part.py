import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.load_data_leaderboard_unet import create_data_loaders_image
from utils.model.varnet_ckp import VarNet
from utils.model.unet_attn import Att_UNet


def test_kspace(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
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
    print('Current cuda device ', torch.cuda.current_device())

    checkpoint1 = torch.load(args.exp_dir / 'best_model_varnet.pt', map_location='cpu')
    checkpoint2 = torch.load(args.exp_dir / 'best_model_unet.tar', map_location='cpu')

    model1 = VarNet(states=checkpoint1['model'], num_cascades=args.cascade, pools=4, chans=18, sens_pools=4, sens_chans=8)
    model1.to(device=device)
    print(checkpoint1['epoch'], checkpoint1['best_val_loss'].item())
    model1.load_state_dict(checkpoint1['model'])

    model2 = Att_UNet(num_class=args.out_chans)
    model2.to(device=device)
    print(checkpoint2['epoch'], checkpoint2['val_loss'].item())
    model2.load_state_dict(checkpoint2['model'])

    forward_loader_kspace = create_data_loaders(data_path=args.data_path_kspace, args=args, isforward=True)
    forward_loader_image = create_data_loaders_image(data_path=args.data_path_image, args=args, isforward=True)
    reconstructions_1, inputs = test_kspace(args, model1, forward_loader_kspace)
    reconstructions_2, inputs = test_image(args, model2, forward_loader_image)

    for fname, _ in reconstructions_1.items():
        reconstructions_1[fname] = reconstructions_1[fname] * 0.93 + reconstructions_2[fname] * 0.07
    # reconstructions = reconstructions_1 * 0.5 + reconstructions_2 * 0.5
    #
    save_reconstructions(reconstructions_1, args.forward_dir, inputs=inputs)