"""DenseASPP + CBAM 完整版U-Net"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class _DenseAsppBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, dilation_rate):
        super(_DenseAsppBlock, self).__init__()
        self.aspp_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels1, 1, bias=False),
            nn.BatchNorm2d(inter_channels1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels1, inter_channels2, 3,
                      dilation=dilation_rate, padding=dilation_rate, bias=False),
            nn.BatchNorm2d(inter_channels2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.aspp_conv(x)


class DenseASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[3, 6, 12, 18]):
        super(DenseASPP, self).__init__()
        self.dilations = dilations
        inter_channels1 = in_channels // 4
        inter_channels2 = in_channels // 2

        self.aspp_blocks = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            self.aspp_blocks.append(
                _DenseAsppBlock(in_channels + i * inter_channels2,
                               inter_channels1, inter_channels2, dilation)
            )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + len(dilations) * inter_channels2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = [x]
        for block in self.aspp_blocks:
            concat_features = torch.cat(features, dim=1)
            out = block(concat_features)
            features.append(out)

        final_concat = torch.cat(features, dim=1)
        return self.conv1x1(final_concat)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownWithDenseASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownWithDenseASPP, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.denseaspp = DenseASPP(in_channels, out_channels)
        self.double_conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.denseaspp(x)
        x = self.double_conv(x)
        return x


class UpWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpWithCBAM, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        self.cbam = CBAM(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.cbam(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DenseASPPCBAMUNet(nn.Module):
    """DenseASPP + CBAM 完整版U-Net"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(DenseASPPCBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownWithDenseASPP(64, 128)
        self.down2 = DownWithDenseASPP(128, 256)
        self.down3 = DownWithDenseASPP(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = DownWithDenseASPP(512, 1024 // factor)

        self.up1 = UpWithCBAM(1024, 512 // factor, bilinear)
        self.up2 = UpWithCBAM(512, 256 // factor, bilinear)
        self.up3 = UpWithCBAM(256, 128 // factor, bilinear)
        self.up4 = UpWithCBAM(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        if self.n_classes == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)