import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.load_data_leaderboard_unet import create_data_loaders_image
from utils.model.dircn.dircn_kinclude import DIRCN
from utils.model.varnet_ckp_kinclude import VarNet
from utils.model import fastmri


def test_kspace(args, model1, model2, data_loader):
    model1.eval()
    model2.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            print(fnames)
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            koutput1 = model1(kspace, mask)
            koutput2 = model2(kspace, mask)
            # print(koutput1.shape) -> torch.Size([1, 16, 768, 396, 2]) 출력
            # koutput = torch.zeros(1, koutput1.shape[1], koutput1.shape[2], koutput1.shape[3], 2)
            # print(koutput.shape)

            koutput1 = koutput1.cpu().numpy()
            koutput2 = koutput2.cpu().numpy()

            koutput = np.zeros((1, koutput1.shape[1], koutput1.shape[2], koutput1.shape[3], 2))

            koutput = 0.5 * koutput1 + 0.5 * koutput2
            print("add")
            koutput[0, :, :, 182:214, :] = 0.5 * koutput1[0, :, :, 182:214, :] + 0.5 * koutput2[0, :, :, 182:214, :]
            
            koutput = torch.Tensor(koutput).cuda()

            # for i in range(koutput1.shape[1]):
            #     print(f"{i}th iteration in {fnames}")
            #     for j in range(koutput1.shape[2]):
            #         for k in range(koutput1.shape[3]):
            #             for l in range(2):
            #                 if 182 <= k <= 213:
            #                     koutput[0][i][j][k][l] = 0.6 * koutput1[0][i][j][k][l] + 0.4 * koutput2[0][i][j][k][l]
            #                 else:
            #                     koutput[0][i][j][k][l] = 0.4 * koutput1[0][i][j][k][l] + 0.6 * koutput2[0][i][j][k][l]

            imoutput = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(koutput)), dim=1)
            height = imoutput.shape[-2]
            width = imoutput.shape[-1]
            imoutput = imoutput[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]
            print(imoutput.shape)

            print(f"{fnames} image 생성 완료")

            for i in range(imoutput.shape[0]):
                print(f"{i}")
                reconstructions[fnames[i]][int(slices[i])] = imoutput[i].cpu().numpy()
            print(f"{fnames} image 저장 완료")

    for fname in reconstructions:
        print(f"{fname} reconstruction 저장을 dict에 담는 중")
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

    checkpoint1 = torch.load(args.exp_dir / 'best_model_dircn.pt')
    checkpoint2 = torch.load(args.exp_dir / 'best_model_varnet.pt')

    model1 = DIRCN(num_cascades=args.cascade1)
    model1.to(device=device)
    print(checkpoint1['epoch'], checkpoint1['best_val_loss'].item())
    model1.load_state_dict(checkpoint1['model'])

    model2 = VarNet(states = checkpoint2['model'], num_cascades=args.cascade2)
    model2.to(device=device)
    print(checkpoint2['epoch'], checkpoint2['best_val_loss'].item())
    model2.load_state_dict(checkpoint2['model'])

    forward_loader_kspace = create_data_loaders(data_path=args.data_path_kspace, args=args, isforward=True)
    torch.save(forward_loader_kspace, args.exp_dir / 'kspace_loader.pt')
    # forward_loader_kspace = torch.load(args.exp_dir / 'kspace_loader.pt')

    reconstructions, inputs = test_kspace(args, model1, model2, forward_loader_kspace)

    print("save reconstruction 시작!")
  
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)

