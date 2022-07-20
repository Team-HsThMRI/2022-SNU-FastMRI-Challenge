import torch
from torch import nn
from torch.nn import functional as F


class Mnet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.first_block = ConvBlock(in_chans, 32)

        self.downshift1 = firstDownShift(32, 64, 0)
        self.downshift2 = DownShift(64, 96, 0)
        self.downshift3 = DownShift(96, 128, 0)
        self.middleshift = MiddleShift(128, 128, 0)
        self.upshift1 = UpShift(128, 96, 0)
        self.upshift2 = UpShift(96, 64, 0)
        self.upshift3 = UpShift(64, 32, 0)

        self.down = Down(32, 32, 0)
        self.down1 = Down(64, 64, 0)
        self.down2 = Down(96, 96, 0)
        self.down3 = Down(128, 128, 0)
        self.up1 = Up(128, 128, 0)
        self.up2 = Up(96, 96, 0)
        self.up3 = Up(64, 64, 0)
        self.rup1 = Up(128, 128, 0)
        self.rup2 = Up(224, 224, 0)
        self.rup3 = Up(288, 288, 0)

        self.last_block = nn.Sequential(
            nn.Conv2d(320, out_chans, kernel_size=1),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(out_chans),
        )

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
        input = input.unsqueeze(1) # 순서대로 batch, channel, height, width

        lleg1 = self.first_block(input)
        lleg2 = self.down(lleg1)
        lleg3 = self.down(lleg2)
        lleg4 = self.down(lleg3)

        dwn1 = self.downshift1(lleg1)
        enc1 = self.down1(dwn1)
        dwn2 = self.downshift2(lleg2, enc1)
        enc2 = self.down2(dwn2)
        dwn3 = self.downshift3(lleg3, enc2)
        enc3 = self.down3(dwn3)
        u1 = self.middleshift(lleg4, enc3)

        rleg1 = u1
        dec1 = self.up1(u1)
        u2 = self.upshift1(dwn3, dec1)
        rleg2 = torch.cat([u2, self.rup1(rleg1)], dim=1)
        dec2 = self.up2(u2)
        u3 = self.upshift2(dwn2, dec2)
        rleg3 = torch.cat([u3, self.rup2(rleg2)], dim=1)
        dec3 = self.up3(u3)
        u4 = self.upshift3(dwn1, dec3)
        rleg4 = torch.cat([u4, self.rup3(rleg3)], dim=1)

        output = self.last_block(rleg4)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class firstDownShift(nn.Module):
    def __init__(self, in_chans, out_chans, p=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.convblock1 = ConvBlock(in_chans, 32)
        self.convblock2 = ConvBlock(64, out_chans)

    def forward(self, x):
        originx = x
        x = self.convblock1(x)
        x = torch.cat([x, originx], dim=1)
        x = self.convblock2(x)
        return x


class DownShift(nn.Module):
    def __init__(self, in_chans, out_chans, p=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.convblock1 = ConvBlock(in_chans + 32, in_chans)
        self.convblock2 = ConvBlock(in_chans * 2, out_chans)

    def forward(self, concat_input, x):
        originx = x
        x = self.convblock1(torch.cat([concat_input, x], dim=1))
        x = self.convblock2(torch.cat([x, originx], dim=1))
        return x


class UpShift(nn.Module):
    def __init__(self, in_chans, out_chans, p=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.convblock1 = ConvBlock(in_chans*2, in_chans)
        self.convblock2 = ConvBlock(in_chans*2, out_chans)

    def forward(self, concat_input, x):
        originx = x
        x = self.convblock1(torch.cat([concat_input, x], dim=1))
        x = self.convblock2(torch.cat([x, originx], dim=1))
        return x


class MiddleShift(nn.Module):
    def __init__(self, in_chans, out_chans, p=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.convblock1 = ConvBlock(in_chans+32, in_chans)
        self.convblock2 = ConvBlock(in_chans*2, in_chans*2)
        self.convblock3 = ConvBlock(in_chans*2, out_chans)

    def forward(self, concat_input, x):
        originx = x
        x = self.convblock1(torch.cat([concat_input, x], dim=1))
        x = self.convblock2(torch.cat([x, originx], dim=1))
        x = self.convblock3(x)
        return x


class Down(nn.Module):
    def __init__(self, in_chans, out_chans, p=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans, p=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.up(x)
        x = self.dropout(x)
        return x