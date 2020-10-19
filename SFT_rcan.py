'''
Architectures for SFT_attention
'''
import torch.nn as nn
import torch.nn.functional as F
import block as B
import common
from common import PAM_Module, CAM_Module
####################
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):

        feat1 = self.conv5a(x)

        sa_feat = self.sa(feat1)


        sa_conv = self.conv51(sa_feat)


        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)


        sc_feat = self.sc(feat2)


        sc_conv = self.conv52(sc_feat)

        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ResBlock_SFT_torch(nn.Module):
    def __init__(self):
        super(ResBlock_SFT_torch, self).__init__()
        self.sft0 = SFTLayer_torch()
        self.conv0 = CALayer(64, 16) ####
        self.sft1 = SFTLayer_torch()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):

        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv0(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class SFTLayer_torch(nn.Module):
    def __init__(self):
        super(SFTLayer_torch, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift

class SFT_Net_torch(nn.Module):
    def __init__(self,args):
        super(SFT_Net_torch, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)
        n_feats = args.n_feats
        kernel_size = 3
        r = args.scale
        reduction=args.reduction
        act = nn.ReLU(True)
        sft_branch = []
        n_resblocks=args.n_resblocks
        for i in range(12):
            sft_branch.append(ResBlock_SFT_torch())

        sft_branch.append(SFTLayer_torch())
        sft_branch.append(ResidualGroup(common.default_conv, n_feats, kernel_size, reduction, act=act, res_scale=args.scale, n_resblocks=n_resblocks))
        sft_branch.append(ResidualGroup(common.default_conv, n_feats, kernel_size, reduction, act=act, res_scale=args.scale, n_resblocks=n_resblocks))
        sft_branch.append(nn.Conv2d(64, 512, 3, 1, 1))
        sft_branch.append(DANetHead(512,64,norm_layer=nn.BatchNorm2d))

        self.sft_branch = nn.Sequential(*sft_branch)

        if r == 2 :
            self.HR_branch = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1)
            ])
        elif r == 4:
            self.HR_branch = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1)
            ])

        elif r==8:
            self.HR_branch = nn.Sequential(*[
                nn.Upsample(scale_factor=4, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1)
            ])

        elif r==16:
            self.HR_branch = nn.Sequential(*[
                nn.Upsample(scale_factor=4, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1)
            ])

        else:
            raise ValueError("scale must be 2 or 4 or 8 or 16.")



        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

        #self.CondNet = nn.Conv2d(3, 128, 1)  ###(input ,output channel, kernel ,padding)
    def forward(self, x):
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res


        out = self.HR_branch(fea)
        return out
