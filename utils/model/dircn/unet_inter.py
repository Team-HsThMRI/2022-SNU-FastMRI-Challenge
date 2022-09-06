import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, List, Optional

class Unet(nn.Module):

    def __init__(self, in_chans, out_chans, n, bias, interconnections, make_interconnections):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.n = n

        self.first_block = ConvBlock(in_chans, n, bias)
        # self.conv1 = ConvBlock_1step(32, 64)
        # self.conv2 = ConvBlock_1step(64, 128)
        # self.conv3 = ConvBlock_1step(128, 128)
        # self.conv4 = ConvBlock_1step(128, 64)
        # self.conv5 = ConvBlock_1step(64, 32)
        # self.last_block = nn.Conv2d(32, out_chans, kernel_size=1)
        self.interconnections = interconnections
        self.make_interconnections = make_interconnections
        
        self.down1 = Down(n, 2*n, bias)
        self.down2 = Down(2*n, 4*n, bias)
        self.down3 = Down(4*n, 8*n, bias)
        self.down4 = Down(8*n, 16*n, bias)
        self.up1_trans = nn.ConvTranspose2d(16*n, 8*n, kernel_size=2, stride=2)
        self.up1 = Up(3*8*n if make_interconnections else 2*8*n, 8*n, bias)

        self.up2_trans = nn.ConvTranspose2d(8*n, 4*n, kernel_size=2, stride=2)
        self.up2 = Up(3*4*n if make_interconnections else 2*4*n, 4*n, bias)

        self.up3_trans = nn.ConvTranspose2d(4*n, 2*n, kernel_size=2, stride=2)
        self.up3 = Up(3*2*n if make_interconnections else 2*2*n, 2*n, bias)

        self.up4_trans = nn.ConvTranspose2d(2*n, n, kernel_size=2, stride=2)
        self.up4 = Up(3*n if make_interconnections else 2*n, n, bias)
        self.last_block = nn.Conv2d(n, out_chans, kernel_size=1)

        self.maxpool = nn.MaxPool2d((12, 5), (12, 5))
        self.fc1 = ULinear(20*16*self.n, 20*16*self.n, 0.5)
        self.invConv = nn.ConvTranspose2d(16*n, 16*n, (12, 5), (12, 5))
        
    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input, internals: Optional[List[torch.Tensor]] = None):
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
        m0 = self.down4(d4)
        # print(m0.shape) # ?x?x48x25
        m1 = self.maxpool(m0)
        # print(m1.shape) # ?x?x4x5
        m2 = m1.view((-1, 20*16*self.n))
        m3 = self.fc1(m2)

        m4 = m3.view(-1, 16 * self.n, 4, 5)
        # print(m4.shape)
        m5 = self.invConv(m4)
        if self.interconnections and internals is not None:
            assert len(internals) == 4, "When using dense cascading, all layers must be given"
            # Connect conv
            d1 = torch.cat([d1, internals[0]], dim=1)
            d2 = torch.cat([d2, internals[1]], dim=1)
            d3 = torch.cat([d3, internals[2]], dim=1)
            d4 = torch.cat([d4, internals[3]], dim=1)
        
        internals = []
        m0 = self.up1_trans(m5)
        u1 = self.up1(m0, d4)
        internals.append(u1)

        u1 = self.up2_trans(u1)
        u2 = self.up2(u1, d3)
        internals.append(u2)

        u2 = self.up3_trans(u2)
        u3 = self.up3(u2, d2)
        internals.append(u3)

        u3 = self.up4_trans(u3)
        u4 = self.up4(u3, d1)
        internals.append(u4)
        
        internals.reverse()
        output = self.last_block(u4)

        # output = self.last_block(c5)
        if self.interconnections:
            return output, internals
        return output

# class ULinear(nn.Module):
#     def __init__(self, in_chans, out_chans, p = 0):
#         super().__init__()
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.layers = nn.Sequential(
#             nn.Linear(in_chans, out_chans, bias = True),
#             nn.BatchNorm1d(out_chans),
#             nn.LeakyReLU(negative_slope = 0.2, inplace=True),
#             nn.Dropout(p)
#         )

#     def forward(self, x):
#         return self.layers(x)

class ULinear(nn.Module):
    def __init__(self, in_chans, out_chans, p = 0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Linear(in_chans, out_chans, bias = True),
            # nn.BatchNorm1d(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.layers(x)

class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans, bias):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans, affine=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans, affine=bias),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans, bias):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans, bias)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans, bias):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv = ConvBlock(in_chans, out_chans, bias)

    def forward(self, x, concat_input):
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)