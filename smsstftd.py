# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Original MS-STFT code was from the "Encodec" implementation, which is licensed under the MIT License, 
# and is available at https://github.com/facebookresearch/encodec

"""MS-STFT discriminator, provided here for reference."""
import typing as tp
import torch
import torchaudio
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import sys

from modules import NormConv2d

FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int]=(1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


def _normalize(tensor, dim):
    denom = tensor.norm(p=2.0, dim=dim, keepdim=True).clamp_min( 1e-12 )
    return tensor / denom

# from https://github.com/sony/bigvsan/blob/main/san_modules.py MIT licensed.
class SANConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None
                 ):
        super(SANConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation,
            groups=1, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        scale = self.weight.norm(p=2.0, dim=[1, 2, 3], keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(in_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, flg_train=False):
        if self.bias is not None:
            input = input + self.bias.view(self.in_channels, 1, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1, 1)
        if flg_train:
            out_fun = F.conv2d(input, normalized_weight.detach(), None, self.stride,
                               self.padding, self.dilation, self.groups)
            out_dir = F.conv2d(input.detach(), normalized_weight, None, self.stride,
                               self.padding, self.dilation, self.groups)
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = F.conv2d(input, normalized_weight, None, self.stride,
                           self.padding, self.dilation, self.groups)
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=[1, 2, 3])

#Slicing STFT Discriminator
class SDiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters_in (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(self,
                 filters_in: int,
                 in_channels: int=1,
                 out_channels: int=1,
                 n_fft: int=1024,
                 hop_length: int=256,
                 win_length: int=1024,
                 max_filters: int=1024,
                 filters_scale: int=1,
                 kernel_size: tp.Tuple[int, int]=(3, 9),
                 dilations: tp.List=[1, 2, 4],
                 stride: tp.Tuple[int, int]=(1, 2),
                 normalized: bool=True,
                 norm: str='weight_norm',
                 activation: str='LeakyReLU',
                 activation_params: dict={'negative_slope': 0.2}):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters_in
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None)
        spec_channels = self.in_channels * 2
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size)))
        in_channels = min(filters_scale * self.filters, max_filters)
        for i, dilate in enumerate(dilations):
            out_channels = min((filters_scale**(i + 1)) * self.filters, max_filters)
            self.convolutions.append(
                NormConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilate, 1),
                    padding=get_2d_padding(kernel_size, (dilate, 1)),
                    norm=norm))
            in_channels = out_channels

        out_channels = min((filters_scale**(len(dilations) + 1)) * self.filters,
                      max_filters)
        self.convolutions.append(
            NormConv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm))
        
        self.conv_post = SANConv2d(
            out_channels,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])))

    def forward(self, x: torch.Tensor, flg_train=False):
        fmap = []
        y = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        y = torch.cat([y.real, y.imag], dim=1)
        y = rearrange(y, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convolutions):
            y = layer(y)
            y = self.activation(y)
            fmap.append(y)
        y = self.conv_post(y, flg_train=flg_train)
        if flg_train:
            y_function, y_direction = y
            fmap.append(y_function)
            y_function = torch.flatten(y_function, 1, -1)
            y_direction = torch.flatten(y_direction, 1, -1)
            z = [y_function, y_direction]
        else:
            fmap.append(y)
            z = torch.flatten(y, 1, -1)

        return z, fmap


class SMultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(self,
                 filters: int,
                 in_channels: int=1,
                 out_channels: int=1,
                 n_ffts: tp.List[int]=[1024, 2048, 512, 256, 128],
                 hop_lengths: tp.List[int]=[256, 512, 128, 64, 32],
                 win_lengths: tp.List[int]=[1024, 2048, 512, 256, 128],
                 **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            SDiscriminatorSTFT(
                filters,
                in_channels=in_channels,
                out_channels=out_channels,
                n_fft=n_ffts[i],
                win_length=win_lengths[i],
                hop_length=hop_lengths[i],
                **kwargs) for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)
    
    def forward(self, y, y_hat, flg_train=False) -> DiscriminatorOutput:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y, flg_train=flg_train)
            y_d_g, fmap_g = d(x=y_hat, flg_train=flg_train)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr_function, dr_direction = dr
        dg_function, dg_direction = dg
        r_loss_fun = torch.mean(F.softplus(1 - dr_function)**2)
        g_loss_fun = torch.mean(F.softplus(dg_function)**2)
        r_loss_dir = torch.mean(F.softplus(1 - dr_direction)**2)
        g_loss_dir = torch.mean(-F.softplus(1 - dg_direction)**2)
        r_loss = r_loss_fun + r_loss_dir
        g_loss = g_loss_fun + g_loss_dir
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(F.softplus(1 - dg)**2)
        gen_losses.append(l)
        loss = loss + l

    return loss, gen_losses