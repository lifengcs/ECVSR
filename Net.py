import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import numpy as np
from common import *
import os
import math
from deform_conv import ModulatedDeformConvPack as DCN


class Motion_Context_Extractor(nn.Module):
    def __init__(self, nf):
        super(Motion_Context_Extractor, self).__init__()
        self.conv_ext1 = nn.Conv2d(1, nf, 5, 1, 2)
        self.conv_ext2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_ext3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_ext4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_ext5 = nn.Conv2d(nf, 2, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_ext1(x))
        feat = self.lrelu(self.conv_ext2(feat))
        feat = self.lrelu(self.conv_ext3(feat))
        feat = self.lrelu(self.conv_ext4(feat))
        feat = self.conv_ext5(feat)
        return feat


class RM(nn.Module):
    def __init__(self, in_ch, nf, out_ch):
        super(RM, self).__init__()
        self.conv1_1 = nn.Conv2d(in_ch, nf // 4, 5, 1, 2)
        self.conv1_2 = nn.Conv2d(nf // 4, nf // 4, 3, 1, 1)
        self.rb1 = ResidualGroup(nf=nf // 4, n_blocks=3)

        self.conv2_1 = nn.Conv2d(nf // 4, nf // 2, 3, 2, 1)
        self.conv2_2 = nn.Conv2d(nf // 2, nf // 2, 3, 1, 1)
        self.rb2 = ResidualGroup(nf=nf // 2, n_blocks=3)

        self.conv3_1 = nn.Conv2d(nf // 2, nf, 3, 2, 1)
        self.conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.rbs = ResidualGroup(nf=nf, n_blocks=3)

        self.dconv4_1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.dconv4_2 = nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1)
        self.dconv4_3 = nn.Conv2d(nf // 2, nf // 2, 3, 1, 1)
        self.rb4 = ResidualGroup(nf=nf // 2, n_blocks=3)
        self.dconv5_1 = nn.ConvTranspose2d(nf // 2, nf // 4, 4, 2, 1)
        self.dconv5_2 = nn.Conv2d(nf // 4, nf // 4, 3, 1, 1)
        self.rb5 = ResidualGroup(nf=nf // 4, n_blocks=3)
        self.dconv6_1 = nn.Conv2d(nf // 4, nf // 4, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf // 4, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        enc_feat1 = self.conv1_2(self.lrelu(self.conv1_1(x)))
        enc_feat1 = self.rb1(enc_feat1)
        enc_feat2 = self.conv2_2(self.lrelu(self.conv2_1(self.lrelu(enc_feat1))))
        enc_feat2 = self.rb2(enc_feat2)
        enc_feat3 = self.conv3_2(self.lrelu(self.conv3_1(self.lrelu(enc_feat2))))
        res_feat = self.rbs(self.lrelu(enc_feat3))
        dec_feat3 = self.dconv4_2(self.lrelu(self.dconv4_1(res_feat)))
        dec_feat3 = self.rb4(self.dconv4_3(self.lrelu(dec_feat3)) + enc_feat2)
        dec_feat2 = self.dconv5_2(self.lrelu(self.dconv5_1(dec_feat3)))
        dec_feat2 = self.rb5(dec_feat2 + enc_feat1)
        dec_feat1 = self.dconv6_1(dec_feat2)
        out = self.conv_last(self.lrelu(dec_feat1))
        return out


class Motion_Align(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(Motion_Align, self).__init__()
        # t-->i
        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dcnpack1 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1,
                            deformable_groups=groups,
                            extra_offset_mask=True)
        # t-->i
        self.offset_conv3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dcnpack2 = DCN(nf, nf, 3, stride=1, padding=1, dilation=1,
                            deformable_groups=groups,
                            extra_offset_mask=True)
        self.conv_1x1 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat, ref_feat):
        feat_fuse = torch.cat([nbr_feat, ref_feat], dim=1)
        offset1 = self.lrelu(self.offset_conv1(feat_fuse))
        offset1 = self.lrelu(self.offset_conv2(offset1))
        nbr_feat_ = self.lrelu(self.dcnpack1([nbr_feat, offset1]))

        offset2 = self.lrelu(self.offset_conv3(feat_fuse))
        offset2 = self.lrelu(self.offset_conv4(offset2))
        ref_feat_ = self.lrelu(self.dcnpack2([ref_feat, offset2]))
        fuse_feat = torch.cat([ref_feat_, nbr_feat_], 1)
        fuse_feat = self.conv_1x1(fuse_feat)

        return fuse_feat


class Conv3DBlock(nn.Module):
    def __init__(self, nf, dilation):
        super(Conv3DBlock, self).__init__()
        self.conv1x3x3 = nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                                   padding=(0, dilation, dilation), dilation=dilation)
        self.conv3x1x1 = nn.Conv3d(nf, nf, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1x3x3(x)
        out = self.conv3x1x1(out)
        out = x + self.lrelu(out)
        return out


class MS3D(nn.Module):
    def __init__(self, nf):
        super(MS3D, self).__init__()
        self.conv_d1 = Conv3DBlock(nf, dilation=1)
        self.conv_d2 = Conv3DBlock(nf, dilation=2)
        self.conv_d3 = Conv3DBlock(nf, dilation=3)
        self.conv_d4 = Conv3DBlock(nf, dilation=4)

        self.conv1x1x1_1 = nn.Conv3d(nf * 2, nf, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv1x1x1_2 = nn.Conv3d(nf * 2, nf, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv1x1x1_3 = nn.Conv3d(nf * 2, nf, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv1x1x1_4 = nn.Conv3d(nf * 4, nf, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x1 = self.conv_d1(x)
        x2 = self.conv_d2(x)
        x2 = self.lrelu(torch.cat([x1, x2], 1))
        x2 = self.conv1x1x1_1(x2)
        x3 = self.conv_d3(x)
        x3 = self.lrelu(torch.cat([x2, x3], 1))
        x3 = self.conv1x1x1_2(x3)
        x4 = self.conv_d4(x)
        x4 = self.lrelu(torch.cat([x3, x4], 1))
        x4 = self.conv1x1x1_3(x4)
        x_fuse = self.lrelu(torch.cat([x1, x2, x3, x4], 1))
        x_fuse = self.conv1x1x1_4(x_fuse)
        x_fuse = x_fuse + x

        return x_fuse


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=True)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale


class Net(nn.Module):
    def __init__(self, cfg, in_nc, out_nc, nf, nframes=7, groups=8, num_blocks=6,
                 center=None):
        super(Net, self).__init__()
        self.center = nframes // 2 if center is None else center
        self.channel = nf
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.feat_extract = ResidualGroup(nf=nf, n_blocks=5)
        self.RBs = ResidualGroup(nf, n_blocks=2)
        self.conv3x3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.align = Motion_Align(nf, groups)
        self.excitation = SpatialGate()
        self.mce = Motion_Context_Extractor(nf)
        self.conv1x3x3 = nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3x1x1 = nn.Conv3d(nf, nf, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        blocks = []
        for _ in range(num_blocks):
            block = MS3D(nf)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        self.fusion = nn.Conv2d(nframes * nf, nf, 1, 1)
        self.feat_Recon = ResidualGroup(nf=nf, n_blocks=5)
        self.up = nn.PixelShuffle(2)
        self.upconv1 = nn.Conv2d(nf, 64 * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, out_nc, 3, 1, 1)
        self.rm = RM(in_ch=in_nc, nf=nf, out_ch=out_nc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        feat = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        
        feat = self.feat_extract(feat)
        feat = feat.view(B, N, -1, H, W)

        ref_feat = feat[:, self.center, :, :, :].clone()
        # alignment
        aligned_feats = []
        pre_excit_feats = []
        for i in range(N):
            nbr_feat = feat[:, i, :, :, :].clone()
            aligned_feat = self.align(nbr_feat, ref_feat)
            pre_excit_feat = nbr_feat
            if i != self.center:
                nbr_frame = x[:, i, :, :, :].clone()
                temp_res = x_center - nbr_frame
                motion_feat = self.mce(temp_res)
                mask = self.excitation(motion_feat)
                aligned_feat = mask * aligned_feat + aligned_feat
            aligned_feats.append(aligned_feat)
            pre_excit_feats.append(pre_excit_feat)
        pre_aligned_feat = list(pre_excit_feats)
        pre_aligned_feat = pre_aligned_feat[3]
        pre_aligned_feat = pre_aligned_feat.view(B, -1, H, W)

        aft_aligned_feat = list(aligned_feats)
        aft_aligned_feat = aligned_feats[3]
        aft_aligned_feat = aft_aligned_feat.view(B, -1, H, W)
        aligned_feats = torch.stack(aligned_feats, dim=1)  # (B, N, C, H, W)
        aligned_feats = aligned_feats.transpose(dim0=1, dim1=2)  # (B, C, N, H, W)
        feat = self.conv1x3x3(aligned_feats)
        feat = self.lrelu(self.conv3x1x1(feat))
        feat = self.middle(feat)
        feat = feat.view(B, -1, H, W)
        feat = self.lrelu(self.fusion(feat))
        feat = self.feat_Recon(feat)

        up_feat = self.upconv1(feat)
        up_feat = self.up(up_feat)
        up_feat = self.upconv2(up_feat)
        up_feat = self.up(up_feat)
        hr = self.conv_last(up_feat)
        sr_out = hr + F.interpolate(x_center, scale_factor=4, mode='bicubic', align_corners=False)
        sr_rf = self.rm(sr_out)

        return sr_out, sr_rf
