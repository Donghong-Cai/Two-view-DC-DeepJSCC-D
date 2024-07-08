# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from GDN import GDN
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding,bias=False)


class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin+1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, snr):
        # out = F.adaptive_avg_pool2d(x, (1,1))
        # out = torch.squeeze(out)
        # out = torch.cat((out, snr), 1)
        if snr.shape[0]>1:
            snr = snr.squeeze()
        snr = snr.unsqueeze(1)
        mu = torch.mean(x, (2, 3))
        out = torch.cat((mu, snr), 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out*x
        return out


class conv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):
        super(conv_ResBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
    def forward(self, x):
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.gdn2(out)
        if self.use_conv1x1 == True:
            x = self.conv3(x)
        out = out+x
        out = self.prelu(out)
        return out


class deconv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_ResBlock, self).__init__()
        self.deconv1 = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)
    def forward(self, x, activate_func='prelu'):
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out+x
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        return out

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##################################################################################################################################


def define_E(norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    net = Encoder((32,8,8),5,256)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    net = Decoder((32,8,8),5,256)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_Dis(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    net = Feature_dis(32)
    return init_net(net, init_type, init_gain, gpu_ids)

class Feature_dis(nn.Module):
    def __init__(self,in_dim):
        super(Feature_dis, self).__init__()
        self.snet1=nn.Sequential(
            nn.Conv2d(in_channels=in_dim,out_channels=in_dim//2,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.snet2=nn.Sequential(
            nn.Conv2d(in_channels=in_dim,out_channels=in_dim//2,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.cnet=nn.Sequential(
            nn.Conv2d(in_channels=in_dim,out_channels=in_dim//2,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )

    def forward(self,input):
        N=input.shape[0]
        s1=self.snet1(input[:,0,...])
        s2=self.snet2(input[:,1,...])
        c1=self.cnet(input[:,0,...])
        c2=self.cnet(input[:,1,...])

        view1=torch.cat((s1,c1),dim=1)
        view2=torch.cat((s2,c2),dim=1)
        dis_feature=torch.stack((view1,view2),dim=1)

        return dis_feature

class Encoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(Encoder, self).__init__()
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv//2
        padding_L = (kernel_sz-1)//2
        self.conv1 = conv_ResBlock(4, Nc_conv, use_conv1x1=True, kernel_size = kernel_sz, stride = 2, padding=padding_L)
        self.conv2 = conv_ResBlock(Nc_conv, Nc_conv, use_conv1x1=True, kernel_size = kernel_sz, stride = 2, padding=padding_L)
        self.conv3 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.conv4 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.conv5 = conv_ResBlock(Nc_conv, enc_N, use_conv1x1=True, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.AF1 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF2 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF3 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF4 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF5 = AF_block(enc_N, enc_N//2, enc_N)
        self.flatten = nn.Flatten()
    def forward(self, x, snr):
        out = self.conv1(x) #(256,16,16)
        out = self.AF1(out, snr)
        out = self.conv2(out)#(256,8,8)
        out = self.AF2(out, snr)
        out_1 = self.conv3(out)#(256,8,8)
        out = self.AF3(out_1, snr)
        out = self.conv4(out)#(256,8,8)
        out = self.AF4(out, snr)
        out = self.conv5(out)#(32,8,8)
        out = self.AF5(out, snr)
        # out = self.flatten(out)
        return out

# The Decoder model with attention feature blocks
class Decoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_deconv):
        super(Decoder, self).__init__()
        self.enc_shape = enc_shape
        Nh_AF1 = enc_shape[0]//2
        Nh_AF = Nc_deconv//2
        padding_L = (kernel_sz-1)//2
        self.deconv1 = deconv_ResBlock(self.enc_shape[0], Nc_deconv, use_deconv1x1=True, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding = 1)
        self.deconv2 = deconv_ResBlock(Nc_deconv, Nc_deconv, use_deconv1x1=True, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding = 1)
        self.deconv3 = deconv_ResBlock(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv4 = deconv_ResBlock(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv5 = deconv_ResBlock(Nc_deconv, 6, use_deconv1x1=True, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.AF1 = AF_block(self.enc_shape[0], Nh_AF1, self.enc_shape[0])
        self.AF2 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF3 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF4 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF5 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
    def forward(self, x, snr):
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = self.AF1(out, snr)
        out = self.deconv1(out)
        out = self.AF2(out, snr)
        out = self.deconv2(out)
        out = self.AF3(out, snr)
        out = self.deconv3(out)
        out = self.AF4(out, snr)
        out = self.deconv4(out)
        out = self.AF5(out, snr)
        out = self.deconv5(out, 'sigmoid')
        return out
