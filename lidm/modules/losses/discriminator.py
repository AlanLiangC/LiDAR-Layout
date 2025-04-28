import functools
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.nn.init import zeros_, kaiming_normal_

from ..basic import ActNorm, CircularConv2d


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class LiDARNLayerDiscriminator(nn.Module):
    """Modified PatchGAN discriminator from Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(LiDARNLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = (4, 4)
        sequence = [CircularConv2d(input_nc, ndf, kernel_size=kw, stride=(1, 2), padding=(1, 2, 1, 2)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1, 2), bias=use_bias, padding=(1, 2, 1, 2)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, bias=use_bias, padding=(1, 2, 1, 2)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            CircularConv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=(1, 2, 1, 2))]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class LiDARNLayerDiscriminatorV2(nn.Module):
    """Modified PatchGAN discriminator from Pix2Pix (larger receptive field)
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(LiDARNLayerDiscriminatorV2, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = (4, 4)
        sequence = [CircularConv2d(input_nc, ndf, kernel_size=kw, stride=(1, 2), padding=(1, 2, 1, 2)), nn.LeakyReLU(0.2, True),
                    CircularConv2d(ndf, ndf, kernel_size=kw, stride=(1, 2), padding=(1, 2, 1, 2)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(2, 2), bias=use_bias, padding=(1, 2, 1, 2)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, bias=use_bias, padding=(1, 2, 1, 2)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            CircularConv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=(1, 2, 1, 2))]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class LiDARNLayerDiscriminatorV3(nn.Module):
    """Modified PatchGAN discriminator from Pix2Pix (larger receptive field)
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(LiDARNLayerDiscriminatorV3, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = (4, 4)
        sequence = [CircularConv2d(input_nc, ndf, kernel_size=(1, 4), stride=(1, 1), padding=(1, 2, 1, 2)), nn.LeakyReLU(0.2, True),
                    CircularConv2d(ndf, ndf, kernel_size=kw, stride=(2, 2), padding=(1, 2, 1, 2)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(2, 2), bias=use_bias, padding=(1, 2, 1, 2)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, bias=use_bias, padding=(1, 2, 1, 2)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            CircularConv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=(1, 2, 1, 2))]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        import pdb; pdb.set_trace()
        return self.main(input)


class PointNetfeat(nn.Module):
    def __init__(self, pts_dim, x=1):
        super(PointNetfeat, self).__init__()
        self.output_channel = int(512 * x)
        self.conv1 = torch.nn.Conv1d(pts_dim, int(64 * x), 1)
        self.conv2 = torch.nn.Conv1d(int(64 * x), int(128 * x), 1)
        self.conv3 = torch.nn.Conv1d(int(128 * x), self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(int(64 * x))
        self.bn2 = nn.BatchNorm1d(int(128 * x))
        self.bn3 = nn.BatchNorm1d(self.output_channel)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # NOTE: should not put a relu
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_channel)
        return x

class PointNet(nn.Module):
    def __init__(self, pts_dim, x, cls_num):
        super(PointNet, self).__init__()
        self.feat = PointNetfeat(pts_dim, x)
        self.fc1 = nn.Linear(512 * x, 256 * x)
        self.fc2 = nn.Linear(256 * x, 256)

        self.pre_bn = nn.BatchNorm1d(pts_dim)
        self.bn1 = nn.BatchNorm1d(256 * x)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        # NOTE: should there put a BN?
        self.fc_c1 = nn.Linear(256, 256)
        self.fc_c2 = nn.Linear(256, cls_num, bias=False)
        self.fc_s1 = nn.Linear(256, 256)
        self.fc_s2 = nn.Linear(256, 1, bias=False)

    def forward(self, x):
        x = self.feat(self.pre_bn(x))
        x = F.relu(self.bn1(self.fc1(x)))
        feat = F.relu(self.bn2(self.fc2(x)))

        x = F.relu(self.fc_c1(feat))
        logits = self.fc_c2(x)

        x = F.relu(self.fc_s1(feat))
        global_logits = self.fc_s2(x)

        return logits, global_logits

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)