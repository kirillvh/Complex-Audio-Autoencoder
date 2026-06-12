import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d
from torch.nn import Conv1d
from torch.nn import Conv2d, Conv3d
from torch.nn import ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm
from c_normalization import CLayerNorm
from c_activation import CReLU, CGeLU
import torch.nn.functional as F
import math
#from APNet2Models import ConvNeXtBlock

import sys
from utils import get_padding
from utils import init_weights

LRELU_SLOPE = 0.1

if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc
    # https://adefossez.github.io/julius/julius/core.html
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)


# This code is adopted from adefossez's julius.lowpass.LowPassFilters
# https://adefossez.github.io/julius/julius/lowpass.html


#return filter [1,1,kernel_size]
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    #ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    time = torch.complex(time, time)
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)
    return filter


class LowPassFilter1d(nn.Module):

    def __init__(self,
                 dim_latent,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)
        # self.conv = nn.Conv1d(dim_latent, dim_latent, kernel_size=kernel_size, padding= self.pad_left, stride=self.stride, groups=dim_latent, dtype=torch.cfloat)
        # self.conv.weight.data = self.filter.expand(dim_latent, -1, -1)

    #input [B,C,T]
    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right),
                      mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1),
                       stride=self.stride, groups=C)
        # out = self.conv(x)
        return out

### Adapted from "Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations"
### https://github.com/hmichaeli/alias_free_convnets
### License: MIT
### Adapted LPF_RFFT class for 1 dimension
# upsample using FFT
def create_recon_rect(N, cutoff=0.5):
    cutoff_low = int((N * cutoff) // 2)
    cutoff_high = int(N - cutoff_low)
    rect_1d = torch.ones(N)
    rect_1d[cutoff_low + 1:cutoff_high] = 0
    if N % 4 == 0:
        # if N is divides by 4, nyquist freq should be 0.5
        # N % 4 =0 means the downsampeled signal is even
        rect_1d[cutoff_low] = 0.5
        rect_1d[cutoff_high] = 0.5
    return rect_1d

def create_lpf_rect(N, cutoff=0.5):
    cutoff_low = int((N * cutoff) // 2)
    cutoff_high = int(N - cutoff_low)
    rect_1d = torch.ones(N)
    rect_1d[cutoff_low + 1:cutoff_high] = 0
    if N % 4 == 0:
        # if N is divides by 4, nyquist freq should be 0
        # N % 4 =0 means the downsampeled signal is even
        rect_1d[cutoff_low] = 0
        rect_1d[cutoff_high] = 0
    return rect_1d

def create_fixed_lpf_rect(N, size):
    rect_1d = torch.ones(N)
    if size < N:
        cutoff_low = size // 2
        cutoff_high = int(N - cutoff_low)
        rect_1d[cutoff_low + 1:cutoff_high] = 0
    return rect_1d

class LPF_RFFT(nn.Module):
    '''
    saves rect in first use
    '''
    def __init__(self, cutoff=0.5, transform_mode='fft', fixed_size=None):
        super(LPF_RFFT, self).__init__()
        self.cutoff = cutoff
        self.fixed_size = fixed_size
        assert transform_mode in ['fft', 'rfft'], f'transform_mode={transform_mode} is not supported'
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft if transform_mode == 'fft' else torch.fft.rfft
        self.itransform = torch.fft.ifft if transform_mode == 'fft' else torch.fft.irfft

    def forward(self, x):
        x_fft = self.transform(x)
        if not hasattr(self, 'rect') or self.rect.shape[0] != x.shape[1]:
            N = x.shape[-1]
            rect = create_lpf_rect(N, self.cutoff) if not self.fixed_size else create_fixed_lpf_rect(N, self.fixed_size)
            rect = rect[:,:int(N/2+1)] if self.transform_mode == 'rfft' else rect
            self.register_buffer('rect', rect)
            self.to(x.device)
        x_fft *= self.rect
        out = self.itransform(x_fft, n=x.shape[-1]) # support odd inputs - need to specify signal size (irfft default is even)
        #out = torch.fft.ifft(x_fft, n=x.shape[-1])

        return out

class LPF_RECON_RFFT(nn.Module):
    '''
        saves rect in first use
        '''
    def __init__(self, cutoff=0.5, transform_mode='fft'):
        super(LPF_RECON_RFFT, self).__init__()
        self.cutoff = cutoff
        assert transform_mode in ['fft', 'rfft'], f'mode={transform_mode} is not supported'
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft if transform_mode == 'fft' else torch.fft.rfft
        self.itransform = (lambda x: (torch.fft.ifft(x))) if transform_mode == 'fft' else torch.fft.irfft


    def forward(self, x):
        x_fft = self.transform(x)
        if not hasattr(self, 'rect') or self.rect.shape[0] != x.shape[1]:
            N = x.shape[-1]
            rect = create_recon_rect(N, self.cutoff)
            rect = rect[:, :int(N / 2 + 1)] if self.transform_mode == 'rfft' else rect
            self.register_buffer('rect', rect)
            self.to(x.device)

        x_fft *= self.rect
        out = self.itransform(x_fft)
        return out


class UpsampleRFFT(nn.Module):
    '''
    input shape is unknown
    '''
    def __init__(self, up=2, transform_mode='fft'):
        super(UpsampleRFFT, self).__init__()
        self.up = up
        self.recon_filter = LPF_RECON_RFFT(cutoff=1 / up, transform_mode=transform_mode)

    def forward(self, x):
        # pad zeros
        batch_size, num_channels, in_height = x.shape
        #in_height = 1
        x = x.reshape([batch_size, num_channels, in_height, 1])
        x = torch.nn.functional.pad(x, [0, self.up - 1, 0, 0])
        x = x.reshape([batch_size, num_channels, in_height * self.up])
        x = self.recon_filter(x) * (self.up ** 2)
        return x


class DownSample1d(nn.Module):

    def __init__(self, dim_latent, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None \
            else kernel_size
        self.lowpass = LowPassFilter1d(dim_latent=dim_latent,
                                       cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)
        return xx
    
class Activation1d(nn.Module):

    def __init__(self,
                 activation,
                 dim_latent,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpsampleRFFT(2)#UpSample1d(dim_latent=dim_latent,ratio=up_ratio, kernel_size=up_kernel_size)
        self.downsample = DownSample1d(dim_latent=dim_latent,ratio=down_ratio, kernel_size=down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.upsample(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)
        return x





class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


    
class STFTResBlock3(nn.Module):
    """ Residual Block for encoding/decoding STFT real/imaginary slices.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim_input, dim_latent=-1, InnerDimMultiplier=4):
        super().__init__()
        if dim_latent == -1:
            dim_latent = dim_input

        self.feature1 = nn.Conv1d(dim_input, dim_latent, kernel_size=7, padding=3, groups=dim_latent, dtype=torch.cfloat) # depthwise conv
        self.feature2 = nn.Conv1d(dim_latent, dim_latent, kernel_size=7, padding=3, groups=dim_latent, dtype=torch.cfloat)
        self.feature3 = nn.Conv1d(dim_input, dim_latent, kernel_size=7, padding=3, groups=dim_latent, dtype=torch.cfloat)
        self.norm_r = LayerNorm(2*dim_latent, eps=1e-6)
        self.norm_c = LayerNorm(2*dim_latent, eps=1e-6)
        self.norm_r2 = LayerNorm(dim_latent, eps=1e-6)
        self.norm_c2 = LayerNorm(dim_latent, eps=1e-6)
        #self.norm = CLayerNorm(input_size=dim, dim=1, eps=1e-6).to('cuda')
        self.pwconv1 = nn.Linear(2*dim_latent, InnerDimMultiplier * dim_latent, dtype=torch.cfloat) # pointwise/1x1 convs, implemented with linear layers
        #self.act = nn.GELU()
        #self.act = CReLU()
        self.act = CGeLU()
        #self.act = Activation1d(dim_latent=InnerDimMultiplier*dim_latent,activation=CGeLU())
        self.upsample = UpsampleRFFT(up=2)#UpSample1d(dim_latent=dim_latent,ratio=up_ratio, kernel_size=up_kernel_size)
        #self.downsample = DownSample1d(dim_latent=dim_latent,ratio=2)
        self.lpf = LPF_RFFT(cutoff=0.5)
        self.grn = GRN(InnerDimMultiplier * dim_latent)
        self.pwconv2 = nn.Linear(InnerDimMultiplier * dim_latent, dim_latent, dtype=torch.cfloat)
        #self.MultType = MultType
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pwconv2.weight.data.copy_(torch.zeros(dim_latent, InnerDimMultiplier * dim_latent))
        self.pwconv2.bias.data.copy_(torch.zeros(dim_latent))
        # self.pwconv1.weight.data.copy_(torch.zeros(InnerDimMultiplier * dim, dim))
        # self.pwconv1.bias.data.copy_(torch.zeros(InnerDimMultiplier*dim))

        # torch.nn.init.dirac_(self.feature1.weight.data, dim)
        # self.feature1.bias.data.copy_(torch.zeros(dim))
        # torch.nn.init.dirac_(self.feature2.weight.data, dim)
        # self.feature2.bias.data.copy_(torch.zeros(dim))

    def forward(self, input, x):
        #=======================================================================================================================
        # input  --> feature1()----->|       |
        #                            | cat() | -->norm1 --> pwconv1() --> act --> grn --> pwconv2()
        # latent --> feature2()----->|-------|                                                |
        #    |------------------------------------------------------------------------------- + --feature3()-->norm2--> latent 
        #=======================================================================================================================
        #input = self.FeedForwardFilter(input)
        #print("STFTResBlock2.input: ", input.shape) #([8, 513, 64])
        #print("STFTResBlock2.input_next: ", input_next.shape) #([8, 513, 64])
        #print("STFTResBlock2.x: ", x.shape) #([8, 513, 64])
        #x = self.feature1(x)
        latent = x
        x = torch.cat((self.feature1(input), self.feature2(x)), dim=1)
        #print("xcnv.shape2: ", x.shape)
        #x = x.permute(0, 2, 1) # (N, C, H, W) -> (N, H, W, C)
        #x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        #print("xcnv.shape3: ", x.shape) #([8, 512, 1026]

        #x = torch.view_as_real(x)
        x = self.norm_r(x.real).type(torch.cfloat) + 1j * self.norm_c(x.imag).type(torch.cfloat)
        #x = torch.view_as_complex(x)
        #print("xcnv.shape4: ", x.shape)
        x = self.pwconv1(x)
        #print("xcnv.shape5: ", x.shape)        
        #if self.training:
        x = x.transpose(1, 2)
        x = self.upsample(x)
        x = x.transpose(1, 2)

        x = self.act(x)
        
        #print("xcnv.shape6: ", x.shape)
        x = self.grn(x) # somehow (B,C,W) -> (1,B,C,W)
        x = x.squeeze(0) # (1,B,C,W) -> (B,C,W) : reverse the extra dim
        #if self.training:
        x = x.transpose(1, 2)
        x = self.lpf(x)
        x = x[:,:,::2]
        x = x.transpose(1, 2)

        #print("xcnv.shape7: ", x.shape)
        #x = x.transpose(1, 2)
        x = self.pwconv2(x)
        #print("xcnv.shape8: ", x.shape)
        #x = x.permute(0, 2, 1)# (N, H, W, C) -> (N, C, H, W)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
            
        #print('? ',x.shape) # ?  torch.Size([8, 2052, 512])
        #print(x.shape) # ([8, 512, 513])
        
        #print("xcnv.shape9: ", x.shape)
        #if self.MultType:
        #x = 2.0*torch.nn.functional.sigmoid(x) # [0,2.0]
        #latent = (latent + x)*x
        # else:
        latent = latent + x

        latent = self.feature3(latent)
        latent = latent.permute(0,2,1)
        latent = self.norm_r2(latent.real).type(torch.cfloat) + 1j * self.norm_c2(latent.imag).type(torch.cfloat)
        latent = latent.permute(0,2,1)
        
        #print("Complex ConvNeXtV2Block output: ", x.shape)
        #x = input + x
        return latent

class ConvNeXtV2Block1Dim(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., InnerDimMultiplier=4):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, dtype=torch.cfloat) # depthwise conv
        self.norm_r = LayerNorm(dim, eps=1e-6)
        self.norm_c = LayerNorm(dim, eps=1e-6)
        #self.norm = CLayerNorm(input_size=dim, dim=1, eps=1e-6).to('cuda')
        self.pwconv1 = nn.Linear(dim, InnerDimMultiplier * dim, dtype=torch.cfloat) # pointwise/1x1 convs, implemented with linear layers
        #self.act = nn.GELU()
        #self.act = CReLU()
        self.act = CGeLU()
        self.grn = GRN(InnerDimMultiplier * dim)
        self.pwconv2 = nn.Linear(InnerDimMultiplier * dim, dim, dtype=torch.cfloat)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pwconv2.weight.data.copy_(torch.zeros(dim, InnerDimMultiplier * dim))
        self.pwconv2.bias.data.copy_(torch.zeros(dim))
        self.pwconv1.weight.data.copy_(torch.zeros(InnerDimMultiplier * dim, dim))
        self.pwconv1.bias.data.copy_(torch.zeros(InnerDimMultiplier*dim))

        torch.nn.init.dirac_(self.dwconv.weight.data, dim)
        self.dwconv.bias.data.copy_(torch.zeros(dim))

    def forward(self, x):
        input = x
        #input = self.FeedForwardFilter(input)
        #print("xcnv.shape1: ", x.shape)
        x = self.dwconv(x)
        #print("xcnv.shape2: ", x.shape)
        #x = x.permute(0, 2, 1) # (N, C, H, W) -> (N, H, W, C)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        #print("xcnv.shape3: ", x.shape)

        #x = torch.view_as_real(x)
        x = self.norm_r(x.real).type(torch.cfloat) + 1j * self.norm_c(x.imag).type(torch.cfloat)
        #x = torch.view_as_complex(x)
        #print("xcnv.shape4: ", x.shape)
        x = self.pwconv1(x)
        #print("xcnv.shape5: ", x.shape)
        x = self.act(x)
        #print("xcnv.shape6: ", x.shape)
        x = self.grn(x) # somehow (B,C,W) -> (1,B,C,W)
        x = x.squeeze(0) # (1,B,C,W) -> (B,C,W) : reverse the extra dim
        #print("xcnv.shape7: ", x.shape)
        x = self.pwconv2(x)
        #print("xcnv.shape8: ", x.shape)
        #x = x.permute(0, 2, 1)# (N, H, W, C) -> (N, C, H, W)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        #print("xcnv.shape9: ", x.shape)
        
        #print("Complex ConvNeXtV2Block output: ", x.shape)
        x = input + x
        return x

    
class STFTEncoder5(torch.nn.Module):
    def __init__(self, h):
        super(STFTEncoder5, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.InputMultiple = 16
        self.OutputMultiple = 4

        #self.Downsample = Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=(4,2), padding=(1,3), dilation=1, dtype=torch.cfloat, device='cuda')
        #self.Downsample = nn.Linear(513, 512, dtype=torch.cfloat)
        
        #self.Downsample.weight.data.copy_(torch.eye(512, 513))
        #self.Downsample.bias.data.copy_(torch.zeros(512))

        #self.Feature1 = Conv1d(in_channels=512, out_channels=512, kernel_size=9, stride=1, padding=4, dilation=1, dtype=torch.cfloat, device='cuda')
        #torch.nn.init.dirac_(self.Feature1.weight.data, 512)
        #self.Feature1.bias.data.copy_(torch.zeros(512))
        #self.Feature2 = Conv1d(in_channels=512, out_channels=512, kernel_size=9, stride=1, padding=4, dilation=1, dtype=torch.cfloat, device='cuda')
        #torch.nn.init.dirac_(self.Feature2.weight.data, 512)
        #self.Feature2.bias.data.copy_(torch.zeros(512))
        #self.Downsample2 = nn.Linear(8*513, 4*513, dtype=torch.cfloat)
        #self.act = torch.nn.functional.gelu()
        #self.EncodeOut = nn.Linear(1026, 1026)

        self.Block1 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block2 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block3 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block4 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block5 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block6 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block7 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block8 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)


    def forward(self, x):
        #print("enc.pre.shape: ", x.shape) # ([8, 1, 513, 64])
        B,_,C,S = x.shape
        # input_next = torch.cat((input_next, input_next[:,:,:,-1:]), dim=3) # extension pad
        # input_next = input_next[:,:,:,-S:] # cut first element to make t+1 data
        # input_next = input_next.squeeze(1)
        # #print("enc.input_next.shape: ", input_next.shape) # ([8, 1, 513, 64])

        x = x.squeeze(1)
        #input_next = x
        input = x
        # x = x.permute(0,2,1)
        # x = self.Downsample(x)
        # x = x.permute(0,2,1)
        #x = self.Feature1(x)
        x = self.Block1(input,x)
        x = self.Block2(input,x)
        x = self.Block3(input,x)
        x = self.Block4(input,x)
        #input = x
        x = self.Block5(input,x)
        x = self.Block6(input,x)
        #input = x
        x = self.Block7(input,x)
        x = self.Block8(input,x)
        #x = self.Feature2(x)
        #print("enc.output.shape: ", x.shape) # ([8, 1, 513, 64])

        #stack for transpose
        x = torch.cat((x.real, x.imag), dim=1)
        x = x.permute(0,2,1) # quantizer transpose
        # x = torch.nn.functional.relu(x)
        # x = self.EncodeOut(x)
        # x = torch.nn.functional.gelu(x)
        return x

class GeneratorC5(torch.nn.Module):
    def __init__(self, h):
        super(GeneratorC5, self).__init__()
        self.h = h
        self.istft = ISTFT(n_fft = h.n_fft, hop_length = h.hop_size, win_length = h.win_size)

        #self.Upsample = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=8, stride=(4,2), padding=(1,3), dilation=1, dtype=torch.cfloat, device='cuda')
        #self.RescaleConv = Conv1d(514, 513, h.ASP_output_conv_kernel_size, 1, padding=get_padding(h.ASP_output_conv_kernel_size, 1), dtype=torch.cfloat, device='cuda')        

        self.Dim = 513
        #self.InputMultiple = 4
        #self.OutputMultiple = 16

        #self.Upsample = nn.Linear(512, 513, dtype=torch.cfloat)
        self.Block8 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block7 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block6 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block5 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block4 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block3 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block2 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)
        self.Block1 = STFTResBlock3(dim_input=513, InnerDimMultiplier=4)

        #self.InAct = CGeLU()
        #self.DecodeIn = nn.Linear(1026, 1026)
        self.DecodeOut = nn.Linear(513, 513, dtype=torch.cfloat)
        #self.DecodeIn = nn.Linear(513, 513, dtype=torch.cfloat)


    def forward(self, x):


        x = x.permute(0,2,1) # undo quantizer transpose
        x = torch.complex(x[:,0:513,:], x[:,513:1026,:])
        x = x.permute(0,2,1)
        #x = self.DecodeIn(x)
        #x = self.InAct(x)
        x = x.permute(0,2,1)
        input = x

        #x = self.Feature2(x)
        x = self.Block8(input,x)
        #input = x
        x = self.Block7(input,x)
        x = self.Block6(input,x)
        #input = x
        x = self.Block5(input,x)
        x = self.Block4(input,x)
        #input = x
        x = self.Block3(input,x)
        x = self.Block2(input,x)
        x = self.Block1(input,x)
        x = x.permute(0,2,1)
        x = self.DecodeOut(x)
        x = x.permute(0,2,1)
        #x = self.Feature1(x)
        # x = x.permute(0,2,1)
        # x = self.Upsample(x)
        audio = self.istft(x) # Normalized=false missing?
        #print("dec.output.shape: ", x.shape) # ([8, 1, 513, 64])
        return x.unsqueeze(1), audio.unsqueeze(1)