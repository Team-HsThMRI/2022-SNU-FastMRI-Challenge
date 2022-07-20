import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.first_block = ConvBlock(in_chans, 64)
        # self.conv1 = ConvBlock_1step(32, 64)
        # self.conv2 = ConvBlock_1step(64, 128)
        # self.conv3 = ConvBlock_1step(128, 128)
        # self.conv4 = ConvBlock_1step(128, 64)
        # self.conv5 = ConvBlock_1step(64, 32)
        # self.last_block = nn.Conv2d(32, out_chans, kernel_size=1)

        self.down1 = Down(64, 128, 0)
        self.down2 = Down(128, 256, 0)
        self.down3 = Down(256, 512, 0)
        self.down4 = Down(512, 1024, 0)
        self.down5 = Down(1024, 2048, 0)
        self.up1 = Up(2048, 1024, 0)
        self.up2 = Up(1024, 512, 0)
        self.up3 = Up(512, 256, 0)
        self.up4 = Up(256, 128, 0)
        self.up5 = Up(128, 64, 0)
        self.last_block = nn.Conv2d(64, out_chans, kernel_size=1)

    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        input, mean, std = self.norm(input)
        input = input.unsqueeze(1)

        # conv_in = self.first_block(input)
        # c1 = self.conv1(conv_in)
        # c2 = self.conv2(c1)
        # c3 = self.conv3(c2)
        # c4 = self.conv4(c3)
        # c5 = self.conv5(c4)

        d1 = self.first_block(input)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        m0 = self.down5(d5)
        u1 = self.up1(m0, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        output = self.last_block(u5)

        # output = self.last_block(c5)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output

class ConvBlock_1step(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans, p):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(p),
            ConvBlock(in_chans, out_chans)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans, p):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)
        self.dropout = nn.Dropout(p)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        concat_output = self.dropout(concat_output)
        return self.conv(concat_output)