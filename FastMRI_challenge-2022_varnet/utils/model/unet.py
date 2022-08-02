"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class modifiedUnet(nn.Module):

    def __init__(self, in_chans, out_chans, chans = 32, num_pool_layers = 4, drop_prob = 0.0,):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.first_block = ConvBlock(in_chans, 64)
        self.down1 = Down(64, 128, 0)
        self.down2 = Down(128, 256, 0)
        self.down3 = Down(256, 512, 0)
        self.down4 = Down(512, 1024, 0)
        self.up1 = Up(1024, 512, 0)
        self.up2 = Up(512, 256, 0)
        self.up3 = Up(256, 128, 0)
        self.up4 = Up(128, 64, 0)
        self.last_block = nn.Conv2d(64, out_chans, kernel_size=1)
        self.maxpool = nn.MaxPool2d(12)
        self.fc1 = ULinear(4096, 4096, 0.5)
        self.invConv = nn.ConvTranspose2d(1024, 1024, 16, 8)


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
        m0 = self.down4(d4)
        # print(m0.shape)
        m1 = self.maxpool(m0)
        # print(m1.shape)
        m2 = m1.view((-1, 4096))
        # print(m2.shape)
        m3 = self.fc1(m2)
        # print(m3.shape)
        m4 = m3.view(-1, 1024, 2, 2)
        # print(m4.shape)
        m5 = self.invConv(m4)
        # print(m5.shape)

        u1 = self.up1(m5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        output = self.last_block(u4)

        # output = self.last_block(c5)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


class ULinear(nn.Module):
    def __init__(self, in_chans, out_chans, p = 0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Linear(in_chans, out_chans, bias = True),
            nn.BatchNorm1d(out_chans),
            nn.ReLU(inplace=True),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.layers(x)

class ConvBlock_1step(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
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
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans, p=0):
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

    def __init__(self, in_chans, out_chans, p=0):
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


# class Unet(nn.Module):
#     """
#     PyTorch implementation of a U-Net model.
#     O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
#     for biomedical image segmentation. In International Conference on Medical
#     image computing and computer-assisted intervention, pages 234–241.
#     Springer, 2015.
#     """
#
#     def __init__(
#         self,
#         in_chans: int,
#         out_chans: int,
#         chans: int = 32,
#         num_pool_layers: int = 4,
#         drop_prob: float = 0.0,
#     ):
#         """
#         Args:
#             in_chans: Number of channels in the input to the U-Net model.
#             out_chans: Number of channels in the output to the U-Net model.
#             chans: Number of output channels of the first convolution layer.
#             num_pool_layers: Number of down-sampling and up-sampling layers.
#             drop_prob: Dropout probability.
#         """
#         super().__init__()
#
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.chans = chans
#         self.num_pool_layers = num_pool_layers
#         self.drop_prob = drop_prob
#
#         self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
#         ch = chans
#         for _ in range(num_pool_layers - 1):
#             self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
#             ch *= 2
#         self.conv = ConvBlock(ch, ch * 2, drop_prob)
#
#         self.up_conv = nn.ModuleList()
#         self.up_transpose_conv = nn.ModuleList()
#         for _ in range(num_pool_layers - 1):
#             self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
#             self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
#             ch //= 2
#
#         self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
#         self.up_conv.append(
#             nn.Sequential(
#                 ConvBlock(ch * 2, ch, drop_prob),
#                 nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
#             )
#         )
#
#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             image: Input 4D tensor of shape `(N, in_chans, H, W)`.
#         Returns:
#             Output tensor of shape `(N, out_chans, H, W)`.
#         """
#
#         stack = []
#         output = image
#
#         # apply down-sampling layers
#         for layer in self.down_sample_layers:
#             output = layer(output)
#             stack.append(output)
#             output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
#
#         output = self.conv(output)
#
#         # apply up-sampling layers
#         for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
#             downsample_layer = stack.pop()
#             output = transpose_conv(output)
#
#             # reflect pad on the right/botton if needed to handle odd input dimensions
#             padding = [0, 0, 0, 0]
#             if output.shape[-1] != downsample_layer.shape[-1]:
#                 padding[1] = 1  # padding right
#             if output.shape[-2] != downsample_layer.shape[-2]:
#                 padding[3] = 1  # padding bottom
#             if torch.sum(torch.tensor(padding)) != 0:
#                 output = F.pad(output, padding, "reflect")
#
#             output = torch.cat([output, downsample_layer], dim=1)
#             output = conv(output)
#
#         return output
#
#
# """
# Facebook Unet Layers
#     ConvBlock
#     TransposeConvBlock
# """
#
#
# class ConvBlock(nn.Module):
#     """
#     A Convolutional Block that consists of two convolution layers each followed by
#     instance normalization, LeakyReLU activation and dropout.
#     """
#
#     def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
#         """
#         Args:
#             in_chans: Number of channels in the input.
#             out_chans: Number of channels in the output.
#             drop_prob: Dropout probability.
#         """
#         super().__init__()
#
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.drop_prob = drop_prob
#
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(out_chans),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Dropout2d(drop_prob),
#             nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm2d(out_chans),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Dropout2d(drop_prob),
#         )
#
#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             image: Input 4D tensor of shape `(N, in_chans, H, W)`.
#         Returns:
#             Output tensor of shape `(N, out_chans, H, W)`.
#         """
#         return self.layers(image)
#
#
# class TransposeConvBlock(nn.Module):
#     """
#     A Transpose Convolutional Block that consists of one convolution transpose
#     layers followed by instance normalization and LeakyReLU activation.
#     """
#
#     def __init__(self, in_chans: int, out_chans: int):
#         """
#         Args:
#             in_chans: Number of channels in the input.
#             out_chans: Number of channels in the output.
#         """
#         super().__init__()
#
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#
#         self.layers = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_chans, out_chans, kernel_size=2, stride=2, bias=False
#             ),
#             nn.InstanceNorm2d(out_chans),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )
#
#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             image: Input 4D tensor of shape `(N, in_chans, H, W)`.
#         Returns:
#             Output tensor of shape `(N, out_chans, H*2, W*2)`.
#         """
#         return self.layers(image)