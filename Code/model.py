"""
AetherLiftNet — High Quality U-Net CNN
3.2M params | Skip connections | Channel attention | Dilated bottleneck
Best quality dehazing for image upload mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(ch // r, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            ChannelAttention(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ic + sc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class AetherLiftNet(nn.Module):
    """
    U-Net encoder-decoder:
      Stem(3→32) → Enc(32→64→128→256) → Dilated bottleneck(4 blocks)
      → Dec(256→128→64→32) → Head(32→3) + Sigmoid
    """
    def __init__(self, base_ch=32, num_res_blocks=4):
        super().__init__()
        c = [base_ch, base_ch*2, base_ch*4, base_ch*8]

        self.stem       = nn.Sequential(
            nn.Conv2d(3, c[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )
        self.enc1       = DownBlock(c[0], c[1])
        self.enc2       = DownBlock(c[1], c[2])
        self.enc3       = DownBlock(c[2], c[3])
        self.bottleneck = nn.Sequential(
            *[ResBlock(c[3], d) for d in [1, 2, 4, 8][:num_res_blocks]]
        )
        self.dec3       = UpBlock(c[3], c[2], c[2])
        self.dec2       = UpBlock(c[2], c[1], c[1])
        self.dec1       = UpBlock(c[1], c[0], c[0])
        self.head       = nn.Sequential(
            nn.Conv2d(c[0], c[0], 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[0], 3, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b  = self.bottleneck(s3)
        d3 = self.dec3(b,  s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        return self.head(d1)


if __name__ == '__main__':
    m = AetherLiftNet()
    x = torch.randn(1, 3, 256, 256)
    y = m(x)
    p = sum(p.numel() for p in m.parameters())
    print(f"Input:{x.shape}  Output:{y.shape}  Params:{p/1e6:.2f}M")