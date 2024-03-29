import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

def maxpool2x2(x):
    mp = nn.MaxPool2d(kernel_size=2, stride=2)
    x = mp(x)
    return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.encoderblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.encoderblock(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CenterBlock, self).__init__()
        mid_channels = int(in_channels*2)

        self.centerblock1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels*2), mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(12),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 2048, bias=True),
            # nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 4096, bias=True),
            # nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )

        self.invconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 16, 8),
        )
        self.centerblock2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.centerblock1(x)
        x = self.maxpool(x)
        x = x.view((-1, 4096))
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1024, 2, 2)
        x = self.invconv(x)
        x = self.centerblock2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        mid_channels = int(in_channels/2)

        self.decoderblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.decoderblock(x)
        return x


class FinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalBlock, self).__init__()
        mid_channels = int(in_channels/2)

        self.finalblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
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
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int),
            )
        self.psi2 = nn.Sequential(
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
        psi = self.relu(psi)
        psi = self.psi2(psi)
        out = x * psi
        return out



class Att_UNet(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        # Encoder part.
        self.encoder1 = EncoderBlock(in_channels=2, out_channels=64)
        self.encoder2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder4 = EncoderBlock(in_channels=256, out_channels=512)
        # Center part.
        self.center = CenterBlock(in_channels=512, out_channels=512)
        # Decoder part.
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        # Final part.
        self.final = FinalBlock(in_channels=128, out_channels=num_class)
        # Attention Gate
        self.att4 = AttentionGates(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionGates(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGates(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGates(F_g=64, F_l=64, F_int=32)

        self.finalconv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def norm(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, x):
        # Encoding, compressive pathway.
        #print(x.shape)
        input, mean, std = self.norm(x)
        #print(x.shape)

        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(input)
        out_endocer1_mp = maxpool2x2(out_encoder1)
        out_encoder2 = self.encoder2(out_endocer1_mp)
        out_endocer2_mp = maxpool2x2(out_encoder2)
        out_encoder3 = self.encoder3(out_endocer2_mp)
        out_endocer3_mp = maxpool2x2(out_encoder3)
        out_encoder4 = self.encoder4(out_endocer3_mp)
        out_endocer4_mp = maxpool2x2(out_encoder4)

        out_center = self.center(out_endocer4_mp)

        out_att4 = self.att4(g=out_center, x=out_encoder4)
        out_decoder4 = self.decoder4(torch.cat((out_center, out_att4), 1))
        out_att3 = self.att3(g=out_decoder4, x=out_encoder3)
        out_decoder3 = self.decoder3(torch.cat((out_decoder4, out_att3), 1))
        out_att2 = self.att2(g=out_decoder3, x=out_encoder2)
        out_decoder2 = self.decoder2(torch.cat((out_decoder3, out_att2), 1))
        out_att1 = self.att1(g=out_decoder2, x=out_encoder1)
        out_final = self.final(torch.cat((out_decoder2, out_att1), 1))
        out_final = self.unnorm(out_final, mean, std)
        out_final = self.finalconv(out_final)
        out_final = out_final.squeeze(1)

        return out_final