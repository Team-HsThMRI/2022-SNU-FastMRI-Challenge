import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

def maxpool2x2(x):
    mp = nn.MaxPool2d(kernel_size=2, stride=2)
    x = mp(x)
    return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(EncoderBlock, self).__init__()

        self.encoderblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=bias),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.encoderblock(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n, bias):
        super(CenterBlock, self).__init__()
        mid_channels = int(in_channels*2)
        self.n = n
        self.maxpool = nn.MaxPool2d((12, 10), (12, 10))

        self.invconv = nn.ConvTranspose2d(8*n, 8*n, (12, 10), (12, 10))

        self.centerblock1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=mid_channels, affine=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels*2), mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=mid_channels, affine=bias),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(40*8*self.n, 40*8*self.n, bias=True),
            # nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
        )

        self.centerblock2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=out_channels, affine=bias),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.centerblock1(x)
        x = self.maxpool(x)
        x = x.view((-1, 40*8*self.n))
        x = self.fc1(x)
        x = x.view(-1, 8 * self.n, 8, 5)
        x = self.invconv(x)
        x = self.centerblock2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(DecoderBlock, self).__init__()
        # mid_channels = int(in_channels/2)

        self.decoderblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels, affine=bias),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.decoderblock(x)
        return x

class DecoderBlock_upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(DecoderBlock_upsample, self).__init__()
        self.decoderblock_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=out_channels, affine=bias),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.decoderblock_upsample(x)
        return x

class FinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(FinalBlock, self).__init__()

        self.finalblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        x = self.finalblock(x)
        return x


class AttentionGates(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGates, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
            )
        self.psi1 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi1(psi)
        out = x * psi
        return out



class Att_UNet(nn.Module):
    def __init__(self, in_chans, out_chans, n, bias, interconnections, make_interconnections):
        super().__init__()
        bias = False
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.n = n
        
        self.interconnections = interconnections
        self.make_interconnections = make_interconnections

        # Encoder part.
        self.encoder1 = EncoderBlock(in_channels=in_chans, out_channels=n, bias=bias)
        self.encoder2 = EncoderBlock(in_channels=n, out_channels=2*n, bias=bias)
        self.encoder3 = EncoderBlock(in_channels=2*n, out_channels=4*n, bias=bias)
        # Center part.
        self.center = CenterBlock(in_channels=4*n, out_channels=4*n, n=n, bias=bias)
        # Decoder convolution part.
        self.decoder3 = DecoderBlock(in_channels=3*4*n if make_interconnections else 2*4*n, out_channels=2*2*n, bias=bias)
        self.decoder2 = DecoderBlock(in_channels=3*2*n if make_interconnections else 2*2*n, out_channels=2*n, bias=bias)
        self.decoder1 = DecoderBlock(in_channels=3*n if make_interconnections else 2*n, out_channels=n, bias=bias)

        # Decoder upsapling part.
        self.decoder_upsample3 = DecoderBlock_upsample(in_channels=2*2*n, out_channels=2*n, bias=bias)
        self.decoder_upsample2 = DecoderBlock_upsample(in_channels=2*n, out_channels=n, bias=bias)
        
        
        # Final part.
        self.final = FinalBlock(n, out_channels=out_chans, bias=bias)
        # Attention Gate
        self.att3 = AttentionGates(F_g=4*n, F_l=4*n, F_int=2*n)
        self.att2 = AttentionGates(F_g=2*n, F_l=2*n, F_int=n)
        self.att1 = AttentionGates(F_g=n, F_l=n, F_int=n//2)

    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, x, internals: Optional[List[torch.Tensor]] = None):
        # Encoding, compressive pathway.
        #print(x.shape)
        #print(x.shape)

        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(x)
        out_endocer1_mp = maxpool2x2(out_encoder1)
        out_encoder2 = self.encoder2(out_endocer1_mp)
        out_endocer2_mp = maxpool2x2(out_encoder2)
        out_encoder3 = self.encoder3(out_endocer2_mp)
        out_endocer3_mp = maxpool2x2(out_encoder3)

        out_center = self.center(out_endocer3_mp)

        new_internals = []
        
        out_att3 = self.att3(g=out_center, x=out_encoder3)
        if self.interconnections and internals is not None:
            assert len(internals) == 3, "When using dense cascading, all layers must be given"
            # Connect conv
            out_att3 = torch.cat([out_att3, internals[2]], dim=1)
        out_decoder3 = self.decoder3(torch.cat((out_center, out_att3), 1))
        new_internals.append(out_decoder3)

        out_decoder3 = self.decoder_upsample3(out_decoder3)
        out_att2 = self.att2(g=out_decoder3, x=out_encoder2)
        if self.interconnections and internals is not None:
            assert len(internals) == 3, "When using dense cascading, all layers must be given"
            # Connect conv
            out_att2 = torch.cat([out_att2, internals[1]], dim=1)
        out_decoder2 = self.decoder2(torch.cat((out_decoder3, out_att2), 1))
        new_internals.append(out_decoder2)

        out_decoder2 = self.decoder_upsample2(out_decoder2)
        out_att1 = self.att1(g=out_decoder2, x=out_encoder1)
        if self.interconnections and internals is not None:
            assert len(internals) == 3, "When using dense cascading, all layers must be given"
            # Connect conv
            out_att1 = torch.cat([out_att1, internals[0]], dim=1)
        out_decoder1 = self.decoder1(torch.cat((out_decoder2, out_att1), 1))
        new_internals.append(out_decoder1)

        new_internals.reverse()
        out_final = self.final(out_decoder1)
        #print(out_final.shape)
        #print(out_final.shape)

        if self.interconnections:
            return out_final, new_internals
        return out_final