import torch
import torch.nn as nn

class ResidualGroup(nn.Module):
    def __init__(self, nf, n_blocks):
        super(ResidualGroup, self).__init__()
        blocks = []
        for _ in range(n_blocks):
            block = ResidualBlock_noBN(nf)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class ReconBlock(nn.Module):
    def __init__(self, nf, out_nc):
        super(ReconBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return out