# try for DSIC-Net
# cuda gpu版 mynet5:暂时可训练； mynet6: 修正bpp过高的GMM问题（归一化问题修复,z1 for y1 && z2+y1 for y2）
import argparse
import math
import random
import time
import shutil
import os
import sys
from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
import numpy as np
import math
import scipy

#在这里改GMM计算方式
from compressai.entropy_models import (EntropyBottleneck, GaussianMixtureConditional, GaussianConditional)
from compressai.layers import GDN, MaskedConv2d
from compressai.layers import *
from compressai.ans import BufferedRansEncoder, RansDecoder  # pylint: disable=E0611,E0401
from compressai.models.utils import update_registered_buffers, conv, deconv

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from PIL import Image

#net defination
# from compressai.models import *
#ywz mynet defination
# from ywz.mytry.mypriors import *

class Enhancement_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.RB1 = ResidualBlock(32, 32)
        self.RB2 = ResidualBlock(32, 32)
        self.RB3 = ResidualBlock(32, 32)
    def forward(self, x):
        identity = x

        out = self.RB1(x)
        out = self.RB2(out)
        out = self.RB3(out)

        out = out + identity
        return out

class Enhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv3x3(3, 32)

        self.EB1 = Enhancement_Block()
        self.EB2 = Enhancement_Block()
        self.EB3 = Enhancement_Block()

        self.conv2 = conv3x3(32, 3)

    def forward(self, x):
        identity = x
        # out = torch.cat((x,x_another_warp),dim=-3)
        out = self.conv1(x)
        out = self.EB1(out)
        out = self.EB2(out)
        out = self.EB3(out)
        out = self.conv2(out)

        out = out + identity
        return out

class Independent_EN(nn.Module):
    def __init__(self):
        super().__init__()

        # Enhancement
        self.EH1 = Enhancement()
        self.EH2 = Enhancement()

    def forward(self, x1_hat,x2_hat):
        # x1_hat_warp = kornia.warp_perspective(x1_hat, h_matrix, (x1_hat.size()[-2], x1_hat.size()[-1]))
        # end decoder
        # print(x1.size())
        # h_inv = torch.inverse(h_matrix)
        # x2_hat_warp = kornia.warp_perspective(x2_hat, h_inv, (x2_hat.size()[-2], x2_hat.size()[-1]))
        # 增强后
        x1_hat1 = self.EH1(x1_hat)
        x2_hat2 = self.EH2(x2_hat)

        return {
            'x1_hat': x1_hat1,
            'x2_hat': x2_hat2
        }

class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """
    def __init__(self, entropy_bottleneck_channels, init_weights=True):
        super().__init__()
        self.entropy_bottleneck1 = EntropyBottleneck(
            entropy_bottleneck_channels)

        self.entropy_bottleneck2 = EntropyBottleneck(
            entropy_bottleneck_channels)


        if init_weights:
            self._initialize_weights()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(m.loss() for m in self.modules()
                       if isinstance(m, EntropyBottleneck))
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args):
        raise NotImplementedError()

    def parameters(self):
        """Returns an iterator over the model parameters."""
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def aux_parameters(self):
        """
        Returns an iterator over the entropy bottleneck(s) parameters for
        the auxiliary loss.
        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            m.update(force=force)

######################################

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp_loss'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'].values())
        out['mse_loss'] = self.mse(output['x_hat'], target) #end to end
        out['loss'] = self.lmbda * 255**2 * out['mse_loss'] + out['bpp_loss']

        return out


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


##########################################################
#y1_hat 生成global_context
class global_context(nn.Module):
    def __init__(self,M,F,C):
        super().__init__()
        self.F = F
        self.F0 = F//3
        self.M = M
        self.C = C
        # 定义新网络 global_context (输入y1)
        self.global_net = nn.Sequential(
            # nn.Conv2d(M,(F*C),kernel_size=5,stride=1,padding=kernel_size // 2) 等价于下行
            conv(M, F * C, kernel_size=5, stride=1),  # padding = kernel_size // 2
            nn.GroupNorm(num_channels=F * C, num_groups=F),
            nn.ReLU(),

            conv(F * C, F * C, kernel_size=5, stride=1),  # padding = kernel_size // 2
            nn.GroupNorm(num_channels=F * C, num_groups=F),
            nn.ReLU(),

            conv(F * C, F * C, kernel_size=5, stride=1),  # padding = kernel_size // 2
            nn.GroupNorm(num_channels=F * C, num_groups=F),
            nn.ReLU(),

            conv(F * C, F * C, kernel_size=5, stride=1),  # padding = kernel_size // 2
        )

    def forward(self,y1):
        temp_y1 = self.global_net(y1)
        #batch_size
        temp3d = torch.reshape(temp_y1, (-1,3, self.F0, self.C, temp_y1.size()[-2], temp_y1.size()[-1])) #增加batch维度
        #reshape to 3d
        return temp3d.split(1,dim=1)  # aa,bb,cc [batch,1,7 , 32,64,48]  F0为3d版的通道 需要进行Conv3d 出来是个tuple(3) 每个维度是[batch,1,F0,C,H/xx,W/xx]

#生成cost_volume
class cost_volume(nn.Module):
    #输出维度为[1,C,H/xx,W/xx] 作为cost volume
    def __init__(self,N,scale_factor,F,C): #N=input_channels,scale_factor:Upsample_factor,
        super(cost_volume, self).__init__()
        self.N = N
        self.scale_factor = scale_factor
        self.F = F
        self.F0 = F // 3
        self.C = C

        self.model1 = nn.Sequential(
            conv(2*self.N,self.N,kernel_size=5,stride=1),
            nn.GroupNorm(num_channels=self.N,num_groups=4),
            nn.ReLU(),

            conv(self.N, self.N, kernel_size=5, stride=1),
            nn.GroupNorm(num_channels=self.N, num_groups=4),
            nn.ReLU(),
        )

        self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.model2 = nn.Sequential(
            #先取 tensor[0]需要4维的 之后再恢复5维，输入到model2里来 #or:torch.squeeze(x)  torch.unsqueeze(x,dim=0)
            # nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv3d(in_channels=self.F0, out_channels=self.F0, kernel_size=5, stride=1, padding=5 // 2),
            nn.GroupNorm(num_channels=self.F0, num_groups=1),
            nn.ReLU(),

            nn.Conv3d(in_channels=self.F0, out_channels=self.F0, kernel_size=5, stride=1, padding=5 // 2),
            nn.GroupNorm(num_channels=self.F0, num_groups=1),
            nn.ReLU(),
            #输出后记得先转2d再concat
        )

        self.model3 = nn.Sequential(
            conv((self.F0 * self.C + self.N), self.N, kernel_size=5, stride=1),
            nn.GroupNorm(num_channels=self.N, num_groups=4),
            nn.ReLU(),

            conv(self.N, self.N, kernel_size=5, stride=1),
            nn.GroupNorm(num_channels=self.N, num_groups=4),
            nn.ReLU(),

            #最后变成C通道，即为disparity
            conv(self.N, self.C, kernel_size=5, stride=1),
            #softmax over dispartiy dimenstion (C is disparity maxnum)
            # nn.functional.softmax(dim=-3),
        )

    def forward(self,h1,h2,d):
        self.h_in = torch.cat((h1,h2),dim=1)
        self.h_out = self.model1(self.h_in)
        #d取输出的tuple其中一个， 然后  [batch_size,1,F0,C,H/xx,W/xx]
        self.d_in = torch.reshape(d,(-1,d.size()[-3],d.size()[-2],d.size()[-1]))#[batch_size*F0,C,H/xx,W/xx]
        self.d_up = self.upsample_layer(self.d_in) #[batch_size*F0,C,H/xx,W/xx]
        self.d_up_3d = torch.reshape(self.d_up,(-1,self.F0,self.C,self.d_up.size()[-2],self.d_up.size()[-1]))   #[batch,F0,C,H/xx,W/xx]
        self.d_out_3d = self.model2(self.d_up_3d) #[batch,F0,C,H/xx,W/xx]
        self.d_out = torch.reshape(self.d_out_3d,(-1,self.F0*self.C,self.d_out_3d.size()[-2],self.d_out_3d.size()[-1])) #[batch,F0*C,H/xx,W/xx]
        #all
        self.all_in = torch.cat((self.h_out,self.d_out),dim=1) #在channel维进行concat
        self.all_out = self.model3(self.all_in)
        #softmax over dispartiy dimenstion (C is disparity maxnum)
        self.cost = nn.functional.softmax(self.all_out,dim=-3)
        return self.cost

# #dense warp
# 修正： 可反向梯度传播
class dense_warp(nn.Module):
    #检查过batch_size  1)
    def __init__(self):
        super().__init__()
    def forward(self,h1,cost):
        g2 = torch.zeros_like(h1)
        #阻断向h1的梯度传播
        clone_h1 = h1.detach() #detach无梯度传播
        #cuda or cpu
        if h1.device.type == 'cuda':
            g2 = g2.to(h1.device.type + ":" + str(h1.device.index))
            clone_h1 = clone_h1.to(h1.device.type + ":" + str(h1.device.index))
        # print("clone_h1_requires_grad")
        # print(clone_h1.requires_grad)
        # print(g2.requires_grad)
        # for d in range(cost.size()[-3]):
        #     g2 += clone_h1.mul(cost[:,d:(d+1),:,:])
        #     # clone_h1[:,:,1:clone_h1.size()[-2]] = clone_h1[:,:,0:(clone_h1.size()[-2]-1)]
        #     # clone_h1[:,:,0:1] = clone_h1[:,:,0:1] - clone_h1[:,:,0:1] #set zeros
        for d in range(cost.size()[-3]): #C,H,W : H是height, 应以W为单位向右平移！
            # print(g2.size())
            # print(cost.size())
            # print(clone_h1.size())
            # print(cost[:, d:(d + 1), :, 0:(cost.size()[-1] - d)].mul(clone_h1[:, :, :, d:cost.size()[-1]]).size())
            # while 1:
            #     pass
            g2[:, :, :, 0:(cost.size()[-1] - d)] += cost[:, d:(d + 1), :, 0:(cost.size()[-1] - d)].mul(clone_h1[:, :, :, d:cost.size()[-1]])
        #运算完毕后
        # print(g2.requires_grad)
        return g2

#Entropy：
#生成z1,z2
class encode_hyper(nn.Module):
    def __init__(self,N,M):
        super().__init__()
        self.encode_hyper = nn.Sequential(
            #先abs 再输入进来
            conv(in_channels=M, out_channels=N, kernel_size=5, stride=1),
            nn.ReLU(),

            conv(in_channels=N, out_channels=N, kernel_size=5), #stride = 2
            nn.ReLU(),

            conv(in_channels=N, out_channels=N, kernel_size=5), #stride = 2
            #出去后接bottleneck
        )
    def forward(self,y):
        self.y_abs = torch.abs(y)
        self.z = self.encode_hyper(self.y_abs)
        return self.z


#空间域池化 兼容不同patch_size
class spatial_pool2d(nn.Module):
    def __init__(self):
        super(spatial_pool2d, self).__init__()
    def forward(self,X):
        # 空间池化
        Y = torch.zeros([X.size()[0], X.size()[1], 1, 1])
        #cuda or cpu!!
        if X.device.type=='cuda':
            Y = Y.to(X.device.type+":"+str(X.device.index))
        for b in range(Y.size()[0]):
            for c in range(Y.size()[1]):
                Y[b, c, 0, 0] = X[b:(b + 1), c:(c + 1), :, :].max()
        return Y

##z1 for y1
class gmm_hyper_y1(nn.Module):
    def __init__(self,N,M,K): #K表示GMM对应正态分布个数
        super().__init__()
        self.N = N
        self.M = M
        self.K = K

        # 每个都是上采样4倍才行
        self.gmm_sigma = nn.Sequential(
            deconv(in_channels=N,out_channels=N,kernel_size=5), #stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(),

            deconv(in_channels=N, out_channels=N, kernel_size=5),# stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.ReLU(),

            conv(in_channels=N,out_channels=(M*K),kernel_size=5,stride=1), #padding=kernel_size//2
            nn.ReLU(),
        )

        self.gmm_means = nn.Sequential(
            deconv(in_channels=N, out_channels=N, kernel_size=5),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(),

            deconv(in_channels=N, out_channels=N, kernel_size=5),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(),

            conv(in_channels=N, out_channels=(M * K), kernel_size=5, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            deconv(in_channels=N, out_channels=N, kernel_size=5),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(),

            deconv(in_channels=N, out_channels=(M*K), kernel_size=5),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
            spatial_pool2d(),
            nn.LeakyReLU(),

            conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
            #出去后要接一个softmax层，表示概率！！
        )

    def forward(self,z1):
        self.sigma = self.gmm_sigma(z1)
        self.means = self.gmm_means(z1)
        #weights要加softmax！！  nn.functional.softmax
        #softmax 加的有问题，需要按照
        # self.weights = nn.functional.softmax(self.gmm_weights(z1),dim = -3)
        #修正
        temp = torch.reshape(self.gmm_weights(z1),(-1,self.K,self.M,1,1)) #方便后续reshape合并时同一个M的数据相邻成组
        temp = nn.functional.softmax(temp,dim=-4)  #每个Mixture进行一次归一
        self.weights = torch.reshape(temp,(-1,self.M*self.K,1,1))
        ###
        return self.sigma,self.means,self.weights

##z2+y1 for y2
class gmm_hyper_y2(nn.Module):
    def __init__(self,N,M,K): #K表示GMM对应正态分布个数
        super().__init__()
        self.N = N
        self.M = M
        self.K = K

        self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=4) #固定为4倍 因为z和y分辨率本身就差4倍

        # 输入与y1分辨率相同，但通道数是2倍
        self.gmm_sigma = nn.Sequential(
            conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
            nn.ReLU(),

            conv(in_channels=N,out_channels=N,kernel_size=5,stride=1),
            nn.ReLU(),

            conv(in_channels=N,out_channels=(M*K),kernel_size=5,stride=1), #padding=kernel_size//2
            nn.ReLU(),
        )

        self.gmm_means = nn.Sequential(
            conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
            nn.LeakyReLU(),

            conv(in_channels=N,out_channels=N,kernel_size=5,stride=1),
            # stride=2,padding=kernel_size//2,output_padding=stride-1
            nn.LeakyReLU(),

            conv(in_channels=N, out_channels=(M * K), kernel_size=5, stride=1),  # padding=kernel_size//2
        )


        self.gmm_weights = nn.Sequential(
            conv(in_channels=(N+M),out_channels=N,kernel_size=5,stride=1),
            nn.LeakyReLU(),

            conv(in_channels=N,out_channels= (M*K),kernel_size=5,stride=1),
            # nn.MaxPool2d(kernel_size=(H//16,W//16)), # ?? 换图像分辨率就要换模型了
            spatial_pool2d(),
            nn.LeakyReLU(),

            conv(in_channels=(M*K), out_channels=(M * K), kernel_size=1, stride=1),  # padding=kernel_size//2
            #出去后要接一个softmax层，表示概率！！
        )

    def forward(self,z2,y1):
        self.up_z2 = self.upsample_layer(z2)
        self.cat_in = torch.cat((self.up_z2,y1),dim=-3)

        self.sigma = self.gmm_sigma(self.cat_in)
        self.means = self.gmm_means(self.cat_in)
        #softmax!!
        # self.weights = nn.functional.softmax(self.gmm_weights(self.cat_in),dim=-3)
        # 修正
        temp = torch.reshape(self.gmm_weights(self.cat_in), (-1, self.K, self.M,  1, 1))  # 方便后续reshape合并时同一个M的数据相邻成组
        temp = nn.functional.softmax(temp, dim=-4)  # 每个Mixture进行一次归一
        self.weights = torch.reshape(temp, (-1, self.M * self.K, 1, 1))
        ###

        return self.sigma,self.means,self.weights

###################################
class Encoder1(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.g_a_conv1 = conv(3, N)
        self.g_a_gdn1 = GDN(N)
        self.g_a_conv2 = conv(N, N)
        self.g_a_gdn2 = GDN(N)
        self.g_a_conv3 = conv(N, N)
        self.g_a_gdn3 = GDN(N)
        self.g_a_conv4 = conv(N, M)

    def forward(self, x):
        # self.y = self.g_a(x)
        self.g_a_c1 = self.g_a_conv1(x) #Tensor
        self.g_a_g1 = self.g_a_gdn1(self.g_a_c1)
        self.g_a_c2 = self.g_a_conv2(self.g_a_g1)  # Tensor
        self.g_a_g2 = self.g_a_gdn2(self.g_a_c2)
        self.g_a_c3 = self.g_a_conv3(self.g_a_g2)  # Tensor
        self.g_a_g3 = self.g_a_gdn3(self.g_a_c3)
        self.g_a_c4 = self.g_a_conv4(self.g_a_g3)  # Tensor
        self.y = self.g_a_c4
        return self.y,self.g_a_g1,self.g_a_g2,self.g_a_g3

class Decoder1(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.g_s_conv1 = deconv(M, N)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.g_s_conv2 = deconv(N, N)
        self.g_s_gdn2 = GDN(N, inverse=True)
        self.g_s_conv3 = deconv(N, N)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.g_s_conv4 = deconv(N, 3)

    def forward(self, y_hat):
        # self.x_hat = self.g_s(self.y_hat)
        self.g_s_c1 = self.g_s_conv1(y_hat)  # Tensor
        self.g_s_g1 = self.g_s_gdn1(self.g_s_c1)
        self.g_s_c2 = self.g_s_conv2(self.g_s_g1)  # Tensor
        self.g_s_g2 = self.g_s_gdn2(self.g_s_c2)
        self.g_s_c3 = self.g_s_conv3(self.g_s_g2)  # Tensor
        self.g_s_g3 = self.g_s_gdn3(self.g_s_c3)
        self.g_s_c4 = self.g_s_conv4(self.g_s_g3)  # Tensor
        self.x_hat = self.g_s_c4
        return self.x_hat,self.g_s_g1,self.g_s_g2,self.g_s_g3

# class Encoder2(nn.Module):
#     def __init__(self, N, M, **kwargs):
#         super().__init__()
#         self.g_a_conv1 = conv(3, N)
#         self.g_a_gdn1 = GDN(N)
#
#         self.g_a_conv2 = conv(2*N, N)
#         self.g_a_gdn2 = GDN(N)
#
#         self.g_a_conv3 = conv(2*N, N)
#         self.g_a_gdn3 = GDN(N)
#
#         self.g_a_conv4 = conv(2*N, M)
#
#     def forward(self, x , g1,g2,g3): #warp input （warp后）
#         # self.y = self.g_a(x)
#         self.g_a_c1 = self.g_a_conv1(x) #Tensor
#         self.g_a_g1 = self.g_a_gdn1(self.g_a_c1)
#
#         self.g_a_c2 = self.g_a_conv2(torch.cat((g1,self.g_a_g1),dim=-3))  # Tensor
#         self.g_a_g2 = self.g_a_gdn2(self.g_a_c2)
#
#         self.g_a_c3 = self.g_a_conv3(torch.cat((g2,self.g_a_g2),dim=-3))  # Tensor
#         self.g_a_g3 = self.g_a_gdn3(self.g_a_c3)
#
#         self.g_a_c4 = self.g_a_conv4(torch.cat((g3,self.g_a_g3),dim=-3))  # Tensor
#         self.y = self.g_a_c4
#         return self.y
#
# class Decoder2(nn.Module):
#     def __init__(self, N, M, **kwargs):
#         super().__init__()
#         self.g_s_conv1 = deconv(M, N)
#         self.g_s_gdn1 = GDN(N, inverse=True)
#
#         self.g_s_conv2 = deconv(2*N, N)
#         self.g_s_gdn2 = GDN(N, inverse=True)
#
#         self.g_s_conv3 = deconv(2*N, N)
#         self.g_s_gdn3 = GDN(N, inverse=True)
#
#         self.g_s_conv4 = deconv(2*N, 3)
#
#     def forward(self, y_hat,g1,g2,g3):
#         # self.x_hat = self.g_s(self.y_hat)
#         self.g_s_c1 = self.g_s_conv1(y_hat)  # Tensor
#         self.g_s_g1 = self.g_s_gdn1(self.g_s_c1)
#
#         self.g_s_c2 = self.g_s_conv2(torch.cat((g1,self.g_s_g1),dim=-3))  # Tensor
#         self.g_s_g2 = self.g_s_gdn2(self.g_s_c2)
#
#         self.g_s_c3 = self.g_s_conv3(torch.cat((g2,self.g_s_g2),dim=-3))  # Tensor
#         self.g_s_g3 = self.g_s_gdn3(self.g_s_c3)
#
#         self.g_s_c4 = self.g_s_conv4(torch.cat((g3,self.g_s_g3),dim=-3))  # Tensor
#         self.x_hat = self.g_s_c4
#         return self.x_hat

###########################################################################


class DSIC(CompressionModel):
    def __init__(self,N=128,M=192,F=21,C=32,K=5,**kwargs): #'cuda:0' or 'cpu'
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        # super(DSIC, self).__init__()
        # self.entropy_bottleneck1 = CompressionModel(entropy_bottleneck_channels=N)
        # self.entropy_bottleneck2 = CompressionModel(entropy_bottleneck_channels=N)
        self.gaussian1 = GaussianMixtureConditional(K = K)
        self.gaussian2 = GaussianMixtureConditional(K = K)
        self.N = int(N)
        self.M = int(M)
        self.F = F
        self.C = C
        self.K = K
        #定义组件
        self.encoder1 = Encoder1(N,M)
        # self.encoder2 = Encoder2(N,M)
        self.decoder1 = Decoder1(N,M)
        # self.decoder2 = Decoder2(N,M)
        # pic2 需要的组件
        self.pic2_g_a_conv1 = conv(3, N)
        self.pic2_g_a_gdn1 = GDN(N)
        self.pic2_g_a_conv2 = conv(2 * N, N)
        self.pic2_g_a_gdn2 = GDN(N)
        self.pic2_g_a_conv3 = conv(2 * N, N)
        self.pic2_g_a_gdn3 = GDN(N)
        self.pic2_g_a_conv4 = conv(2 * N, M)
        #
        self.pic2_g_s_conv1 = deconv(M, N)
        self.pic2_g_s_gdn1 = GDN(N, inverse=True)
        self.pic2_g_s_conv2 = deconv(2 * N, N)
        self.pic2_g_s_gdn2 = GDN(N, inverse=True)
        self.pic2_g_s_conv3 = deconv(2 * N, N)
        self.pic2_g_s_gdn3 = GDN(N, inverse=True)
        self.pic2_g_s_conv4 = deconv(2 * N, 3)
        #end of pic2
        #######
        self._global_context = global_context(M,F,C)

        #scale_factor 超分辨几倍 (from H,W/16)
        self._cost_volume1 = cost_volume(N,8,F,C) #最外层
        self._cost_volume2 = cost_volume(N,4, F, C)
        self._cost_volume3 = cost_volume(N,2, F, C)#最里层
        self._cost_volume4 = cost_volume(N, 2, F, C)  # 最里层
        self._cost_volume5 = cost_volume(N, 4, F, C)
        self._cost_volume6 = cost_volume(N, 8, F, C)  # 最外层

        self._warp1 = dense_warp()
        self._warp2 = dense_warp()
        self._warp3 = dense_warp()
        self._warp4 = dense_warp()
        self._warp5 = dense_warp()
        self._warp6 = dense_warp()

        #hyper
        self._h_a1 = encode_hyper(N=N,M=M)
        self._h_a2 = encode_hyper(N=N,M=M)
        self._h_s1 = gmm_hyper_y1(N=N,M=M,K=K)
        self._h_s2 = gmm_hyper_y2(N=N,M=M,K=K)

    def forward(self,x1,x2):
        #定义结构
        y1,g1_1,g1_2,g1_3 = self.encoder1(x1)
        z1 = self._h_a1(y1)
        #print(z1.device)
        z1_hat,z1_likelihoods = self.entropy_bottleneck1(z1)
        gmm1 = self._h_s1(z1_hat) #三要素
        y1_hat, y1_likelihoods = self.gaussian1(y1, gmm1[0],gmm1[1],gmm1[2])  # sigma

        x1_hat,g1_4,g1_5,g1_6 = self.decoder1(y1_hat)

        #############################################
        y1_global_context = self._global_context(y1_hat)  # 用y1_hat 看补充材料的图注说明  #返回tuple(3)
        #encoder
        #warp+pic2
        #pic2_1
        pic2_g_a_c1 = self.pic2_g_a_conv1(x2)  # Tensor
        pic2_g_a_g1 = self.pic2_g_a_gdn1(pic2_g_a_c1)
        #end pic2_1

        # pic2_2
        y1_cost_1 = self._cost_volume1(g1_1,pic2_g_a_g1,y1_global_context[0]) #最外层
        y1_warp_1 = self._warp1(g1_1,y1_cost_1)
        pic2_g_a_c2 = self.pic2_g_a_conv2(torch.cat((y1_warp_1,pic2_g_a_g1),dim=-3))
        pic2_g_a_g2 = self.pic2_g_a_gdn2(pic2_g_a_c2)
        #end pic2_2

        # pic2_3
        y1_cost_2 = self._cost_volume2(g1_2, pic2_g_a_g2, y1_global_context[1])
        y1_warp_2 = self._warp2(g1_2, y1_cost_2)
        pic2_g_a_c3 = self.pic2_g_a_conv3(torch.cat((y1_warp_2, pic2_g_a_g2), dim=-3))
        pic2_g_a_g3 = self.pic2_g_a_gdn3(pic2_g_a_c3)
        # end pic2_3

        # pic2_4
        y1_cost_3 = self._cost_volume3(g1_3, pic2_g_a_g3, y1_global_context[2])
        y1_warp_3 = self._warp3(g1_3, y1_cost_3)
        y2 = self.pic2_g_a_conv4(torch.cat((y1_warp_3, pic2_g_a_g3), dim=-3)) #y2 = pic2_g_a_c4
        # end pic2_4
        #end encoder

        # hyper for pic2
        z2 = self._h_a2(y2)
        z2_hat, z2_likelihoods = self.entropy_bottleneck2(z2)
        gmm2 = self._h_s2(z2_hat, y1_hat)  # 三要素
        # 临时
        y2_hat, y2_likelihoods = self.gaussian2(y2, gmm2[0], gmm2[1], gmm2[2])  # 这里也是临时，待改gmm
        # end hyper for pic2

        ##decoder
        #pic2_1
        pic2_g_s_c1 = self.pic2_g_s_conv1(y2_hat)  # Tensor
        pic2_g_s_g1 = self.pic2_g_s_gdn1(pic2_g_s_c1)
        #end pic2_1

        #pic2_2
        y1_cost_4 = self._cost_volume4(g1_4, pic2_g_s_g1, y1_global_context[2])
        y1_warp_4 = self._warp4(g1_4, y1_cost_4)
        pic2_g_s_c2 = self.pic2_g_s_conv2(torch.cat((y1_warp_4, pic2_g_s_g1), dim=-3))  # Tensor
        pic2_g_s_g2 = self.pic2_g_s_gdn2(pic2_g_s_c2)
        #end pic2_2

        #pic2_3
        y1_cost_5 = self._cost_volume5(g1_5, pic2_g_s_g2, y1_global_context[1])
        y1_warp_5 = self._warp5(g1_5, y1_cost_5)
        pic2_g_s_c3 = self.pic2_g_s_conv3(torch.cat((y1_warp_5, pic2_g_s_g2), dim=-3))  # Tensor
        pic2_g_s_g3 = self.pic2_g_s_gdn3(pic2_g_s_c3)
        #end pic2_3

        #pic2_4
        y1_cost_6 = self._cost_volume6(g1_6, pic2_g_s_g3, y1_global_context[0])
        y1_warp_6 = self._warp6(g1_6, y1_cost_6)
        x2_hat = self.pic2_g_s_conv4(torch.cat((y1_warp_6, pic2_g_s_g3), dim=-3))  # Tensor #即为x2_hat = pic2_g_s_c4
        #end pic2_4
        #end decoder
        # print(x1.size())

        return {
            'x1_hat': x1_hat,
            'x2_hat': x2_hat,
            'likelihoods':{
                'y1': y1_likelihoods,
                'y2': y2_likelihoods,
                'z1': z1_likelihoods,
                'z2': z2_likelihoods,
            }
        }
    #quantize
    def _quantize(self, inputs, mode, means=None):
        # type: (Tensor, str, Optional[Tensor]) -> Tensor
        if mode not in ('noise', 'dequantize', 'symbols'):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == 'noise':
            if torch.jit.is_scripting():
                half = float(0.5)
                noise = torch.empty_like(inputs).uniform_(-half, half)
            else:
                noise = self._get_noise_cached(inputs)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == 'dequantize':
            if means is not None:
                outputs += means
            return outputs

        assert mode == 'symbols', mode
        outputs = outputs.int()
        return outputs
    #erfc
    def _standardized_cumulative(self, inputs):
        # type: (Tensor) -> Tensor
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)
    #compress
    def compress(self,x1,x2,output_name,output_path=""):
        torch.cuda.synchronize()
        
        #定义结构
        y1,g1_1,g1_2,g1_3 = self.encoder1(x1)
        z1 = self._h_a1(y1)
        #print(z1.device)
        # z1_hat,z1_likelihoods = self.entropy_bottleneck1(z1)
        # compress
        z1_strings = self.entropy_bottleneck1.compress(z1)
        z1_hat = self.entropy_bottleneck1.decompress(z1_strings, z1.size()[-2:])  # z解码后结果（压缩时仍需要）
        
        gmm1 = self._h_s1(z1_hat) #三要素
        #
        # y1_hat, y1_likelihoods = self.gaussian1(y1, gmm1[0],gmm1[1],gmm1[2])  # sigma
        y1_hat = self._quantize(y1,'dequantize',means=None)
        #
        x1_hat,g1_4,g1_5,g1_6 = self.decoder1(y1_hat)

        #############################################
        y1_global_context = self._global_context(y1_hat)  # 用y1_hat 看补充材料的图注说明  #返回tuple(3)
        #encoder
        #warp+pic2
        #pic2_1
        pic2_g_a_c1 = self.pic2_g_a_conv1(x2)  # Tensor
        pic2_g_a_g1 = self.pic2_g_a_gdn1(pic2_g_a_c1)
        #end pic2_1

        # pic2_2
        y1_cost_1 = self._cost_volume1(g1_1,pic2_g_a_g1,y1_global_context[0]) #最外层
        y1_warp_1 = self._warp1(g1_1,y1_cost_1)
        pic2_g_a_c2 = self.pic2_g_a_conv2(torch.cat((y1_warp_1,pic2_g_a_g1),dim=-3))
        pic2_g_a_g2 = self.pic2_g_a_gdn2(pic2_g_a_c2)
        #end pic2_2

        # pic2_3
        y1_cost_2 = self._cost_volume2(g1_2, pic2_g_a_g2, y1_global_context[1])
        y1_warp_2 = self._warp2(g1_2, y1_cost_2)
        pic2_g_a_c3 = self.pic2_g_a_conv3(torch.cat((y1_warp_2, pic2_g_a_g2), dim=-3))
        pic2_g_a_g3 = self.pic2_g_a_gdn3(pic2_g_a_c3)
        # end pic2_3

        # pic2_4
        y1_cost_3 = self._cost_volume3(g1_3, pic2_g_a_g3, y1_global_context[2])
        y1_warp_3 = self._warp3(g1_3, y1_cost_3)
        y2 = self.pic2_g_a_conv4(torch.cat((y1_warp_3, pic2_g_a_g3), dim=-3)) #y2 = pic2_g_a_c4
        # end pic2_4
        #end encoder

        # hyper for pic2
        z2 = self._h_a2(y2)
        # z2_hat, z2_likelihoods = self.entropy_bottleneck2(z2)
        z2_strings = self.entropy_bottleneck2.compress(z2)
        z2_hat = self.entropy_bottleneck2.decompress(z2_strings, z2.size()[-2:])  # z解码后结果（压缩时仍需要）

        gmm2 = self._h_s2(z2_hat, y1_hat)  # 三要素
        # 临时
        # y2_hat, y2_likelihoods = self.gaussian2(y2, gmm2[0], gmm2[1], gmm2[2])  # 这里也是临时，待改gmm
        y2_hat = self._quantize(y2, 'dequantize', means=None)
        # end hyper for pic2


        ##encoding
        y1_hat_cpu_np = y1_hat.cpu().numpy().astype('int')
        y2_hat_cpu_np = y2_hat.cpu().numpy().astype('int')
        # print(y1_hat)
        # print(y1_hat_cpu_np)
        #========debug
        # print(z1_strings)
        # print(len(z1_strings))
        # print(z1_strings[0].shape)

        #========encoding for header of z1 and z2
        output1 = os.path.join(output_path,str(output_name)+".npz")
        if os.path.exists(output1):
            os.remove(output1)
        fileobj = open(output1, mode='wb')
        # 图片尺寸信息
        fileobj.write(np.array(x1.shape[2:], dtype=np.uint16).tobytes()) #4bytes 512x512 
        # 非零通道号信息
        flag1 = np.zeros(y1_hat.shape[1], dtype=np.int)
        flag2 = np.zeros(y2_hat.shape[1], dtype=np.int)
        for ch_idx in range(y1_hat.shape[1]):
            if np.sum(abs(y1_hat_cpu_np[:, ch_idx,:,:])) > 0:
                flag1[ch_idx] = 1
        non_zero_idx_1 = np.squeeze(np.where(flag1 == 1))
        num1 = np.packbits(np.reshape(flag1, [8, y1_hat.shape[1]//8]))
        minmax1 = np.maximum(abs(y1_hat_cpu_np.max()), abs(y1_hat_cpu_np.min()))
        minmax1 = int(np.maximum(minmax1, 1))
        ###
        for ch_idx in range(y2_hat.shape[1]):
            if np.sum(abs(y2_hat_cpu_np[:, ch_idx,:,:])) > 0:
                flag2[ch_idx] = 1
        non_zero_idx_2 = np.squeeze(np.where(flag2 == 1))
        num2 = np.packbits(np.reshape(flag2, [8, y2_hat.shape[1]//8]))
        minmax2 = np.maximum(abs(y2_hat_cpu_np.max()), abs(y2_hat_cpu_np.min()))
        minmax2 = int(np.maximum(minmax2, 1))
        ########
        fileobj.write(np.array([len(z1_strings[0]), minmax1], dtype=np.uint16).tobytes()) #4bytes
        fileobj.write(np.array(num1, dtype=np.uint8).tobytes()) #y_hat_channels_number // 8  eg:192//8=24
        fileobj.write(z1_strings[0]) # 按照len(z1_strings)来
        #
        fileobj.write(np.array([len(z2_strings[0]), minmax2], dtype=np.uint16).tobytes()) #4bytes
        fileobj.write(np.array(num2, dtype=np.uint8).tobytes()) #y_hat_channels_number // 8  eg:192//8=24
        fileobj.write(z2_strings[0]) # 按照len(z2_strings)来

        fileobj.close()
        #========encoding for range coder of y1 and y2
        # 逐像素计算概率分布、进行熵编码->后续考虑加入joint 进一步提高
        output2 = os.path.join(output_path,str(output_name)+ '.bin')
        if os.path.exists(output2):
            os.remove(output2)
        encoder = RangeEncoder(output2)
        # TINY = 1e-10
        # samples1 = np.arange(0, minmax1*2+1)
        # samples1 = torch.Tensor(samples1).to('cuda:0')
        samples1_np = np.arange(0, minmax1*2+1)
        samples2_np = np.arange(0, minmax2*2+1)

        start = time.time()

        # #y1
        # for h_idx in range(y1_hat_cpu_np.shape[2]):
        #     for w_idx in range(y1_hat_cpu_np.shape[3]):
        #         #如果是joint一类，在此计算当前像素点的概率
        #         for i in range(len(non_zero_idx_1)):
        #             ch_idx = non_zero_idx_1[i] #非零通道
                    
        #             #gmm1 = [scales, means,weights] 通道数均为M*K
        #             ch_idx_list = [ch_idx+temp_i*self.M for temp_i in range(self.K)]
        #             sigma1 = gmm1[0][0,ch_idx_list,h_idx,w_idx]
        #             means1 = gmm1[1][0,ch_idx_list,h_idx,w_idx] + minmax1  #注意这里~~~ 相当于平移
        #             weights1 = gmm1[2][0,ch_idx_list,0,0] #(M*k):((k+1)*M) #1x1
        #             #torch.Size([5])
        #             # print(sigma1.shape)
        #             # print(means1.shape)
        #             # print(weights1.shape)
                    
        #             #计算pmf
        #             pmf = None
        #             for temp_K in range(self.K):
        #                 half = float(0.5)

        #                 values = samples1 - means1[temp_K]

        #                 temp_scales = self.gaussian1.lower_bound_scale(sigma1[temp_K])  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

        #                 values = abs(values)
        #                 upper = self._standardized_cumulative((half - values) / temp_scales)
        #                 lower = self._standardized_cumulative((-half - values) / temp_scales)
        #                 if pmf==None:
        #                     pmf = (upper - lower)* weights1[temp_K] #指定对应的M个通道
        #                 else:
        #                     pmf += (upper - lower)* weights1[temp_K] #指定对应的M个通道

        #             # print(pmf.shape)
        #             # print("minmax1",minmax1)
                    
        #             pmf = pmf.cpu().numpy()
        #             # To avoid the zero-probability            
        #             pmf_clip = np.clip(pmf, 1.0/65536, 1.0)
        #             pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
        #             cdf = list(np.add.accumulate(pmf_clip))
        #             cdf = [0] + [int(i) for i in cdf]
                            
        #             symbol = np.int(y1_hat[0,ch_idx,h_idx,w_idx] + minmax1)
        #             # print("len-cdf",len(cdf))
        #             # raise ValueError("stop")
        #             encoder.encode([symbol], cdf)
        # #end y1

        # y1
        # for h_idx in range(y2_hat_cpu_np.shape[2]):
        #     for w_idx in range(y2_hat_cpu_np.shape[3]):
                # 如果是joint一类，在此计算当前像素点的概率
        for i in range(len(non_zero_idx_1)):
            samples1 = torch.Tensor(samples1_np).to('cuda:0')
            samples1 = samples1.reshape([samples1.shape[0],1,1])
            samples1 = samples1.expand(samples1.shape[0],y1_hat_cpu_np.shape[2],y1_hat_cpu_np.shape[3]) #[arange(xx),width,height]
            #非零通道：一个通道算一次-> 并行优化
            ch_idx = non_zero_idx_1[i] #非零通道
            
            #gmm1 = [scales, means,weights] 通道数均为M*K
            ch_idx_list = [ch_idx+temp_i*self.M for temp_i in range(self.K)]
            sigma1 = gmm1[0][0,ch_idx_list,:,:]
            means1 = gmm1[1][0,ch_idx_list,:,:] + minmax1  #注意这里~~~ 相当于平移
            weights1 = gmm1[2][0,ch_idx_list,0,0] #(M*k):((k+1)*M) #1x1
            #torch.Size([5])
            # print(sigma1.shape)
            # print(means1.shape)
            # print(weights1.shape)
            
            #计算pmf
            pmf = None
            for temp_K in range(self.K):
                half = float(0.5)

                values = samples1 - means1[temp_K]

                temp_scales = self.gaussian1.lower_bound_scale(sigma1[temp_K])  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

                values = abs(values)
                upper = self._standardized_cumulative((half - values) / temp_scales)
                lower = self._standardized_cumulative((-half - values) / temp_scales)
                if pmf==None:
                    pmf = (upper - lower)* weights1[temp_K] #指定对应的M个通道
                else:
                    pmf += (upper - lower)* weights1[temp_K] #指定对应的M个通道

            # print(pmf.shape)
            # raise ValueError("stop")
            # print("minmax1",minmax1)
            
            pmf = pmf.cpu().numpy()
            # print(pmf.shape)
            # raise ValueError("stop")

            for h_idx in range(y1_hat_cpu_np.shape[2]):
                for w_idx in range(y1_hat_cpu_np.shape[3]):
                    pmf_temp = pmf[:,h_idx,w_idx]
                    # print(pmf_temp.shape)
                    # raise ValueError("stop")
                    
                    # To avoid the zero-probability            
                    pmf_clip = np.clip(pmf_temp, 1.0/65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(i) for i in cdf]
                            
                    symbol = np.int(y1_hat[0,ch_idx,h_idx,w_idx] + minmax1)
                    # print("len-cdf",len(cdf))
                    # raise ValueError("stop")
                    encoder.encode([symbol], cdf)
                
        # end y1


        #y2
        
        # for h_idx in range(y2_hat_cpu_np.shape[2]):
        #     for w_idx in range(y2_hat_cpu_np.shape[3]):
                # 如果是joint一类，在此计算当前像素点的概率
        for i in range(len(non_zero_idx_2)):
            samples2 = torch.Tensor(samples2_np).to('cuda:0')
            samples2 = samples2.reshape([samples2.shape[0],1,1])
            samples2 = samples2.expand(samples2.shape[0],y2_hat_cpu_np.shape[2],y2_hat_cpu_np.shape[3]) #[arange(xx),width,height]
            #非零通道：一个通道算一次-> 并行优化
            ch_idx = non_zero_idx_2[i] #非零通道
            
            #gmm2 = [scales, means,weights] 通道数均为M*K
            ch_idx_list = [ch_idx+temp_i*self.M for temp_i in range(self.K)]
            sigma2 = gmm2[0][0,ch_idx_list,:,:]
            means2 = gmm2[1][0,ch_idx_list,:,:] + minmax2  #注意这里~~~ 相当于平移
            weights2 = gmm2[2][0,ch_idx_list,0,0] #(M*k):((k+1)*M) #1x1
            #torch.Size([5])
            # print(sigma2.shape)
            # print(means2.shape)
            # print(weights2.shape)
            
            #计算pmf
            pmf = None
            for temp_K in range(self.K):
                half = float(0.5)

                values = samples2 - means2[temp_K]

                temp_scales = self.gaussian2.lower_bound_scale(sigma2[temp_K])  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

                values = abs(values)
                upper = self._standardized_cumulative((half - values) / temp_scales)
                lower = self._standardized_cumulative((-half - values) / temp_scales)
                if pmf==None:
                    pmf = (upper - lower)* weights2[temp_K] #指定对应的M个通道
                else:
                    pmf += (upper - lower)* weights2[temp_K] #指定对应的M个通道

            # print(pmf.shape)
            # raise ValueError("stop")
            # print("minmax2",minmax2)
            
            pmf = pmf.cpu().numpy()
            # print(pmf.shape)
            # raise ValueError("stop")

            for h_idx in range(y2_hat_cpu_np.shape[2]):
                for w_idx in range(y2_hat_cpu_np.shape[3]):
                    pmf_temp = pmf[:,h_idx,w_idx]
                    # print(pmf_temp.shape)
                    # raise ValueError("stop")
                    
                    # To avoid the zero-probability            
                    pmf_clip = np.clip(pmf_temp, 1.0/65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(i) for i in cdf]
                            
                    symbol = np.int(y2_hat[0,ch_idx,h_idx,w_idx] + minmax2)
                    # print("len-cdf",len(cdf))
                    # raise ValueError("stop")
                    encoder.encode([symbol], cdf)
                
        
        #end y2
        ##end encoding
        encoder.close()
        end = time.time()
        
        num_pixels = x1.shape[2]*x1.shape[3] * 2 # 双目两张图
        size_real = os.path.getsize(output1) + os.path.getsize(output2)

        bpp_real = (os.path.getsize(output1) + os.path.getsize(output2))* 8 / num_pixels #主要看这个
        bpp_side = (os.path.getsize(output1))* 8 / num_pixels

        # print("Time : {:0.3f}".format(end-start))
        print("bpp_real:",bpp_real)

        return {
            'bpp_real': bpp_real,
            'time':None,
             
             'y1_hat':y1_hat,
            'y2_hat':y2_hat,
            'z1_hat':z1_hat,
            'z2_hat':z2_hat,

        }

    #decompress
    def decompress(self,device,output_name,output_path=""):
        output1 = os.path.join(output_path,str(output_name)+ '.npz')
        output2 = os.path.join(output_path,str(output_name)+ '.bin')
        
        #=================decoding for z1 and z2
        fileobj = open(output1, mode='rb')
        x_shape = np.frombuffer(fileobj.read(4), dtype=np.uint16) #图片尺寸 [width,height] array([512, 512], dtype=uint16)
        #z1
        length1, minmax1 = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        num1 = np.frombuffer(fileobj.read(self.M//8), dtype=np.uint8)
        string1 = fileobj.read(length1)
        #z2 
        length2, minmax2 = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        num2 = np.frombuffer(fileobj.read(self.M//8), dtype=np.uint8) #192//8=24
        string2 = fileobj.read(length2)
        fileobj.close()

        flag1 = np.unpackbits(num1)
        non_zero_idx_1 = np.squeeze(np.where(flag1 == 1))
        flag2 = np.unpackbits(num2)
        non_zero_idx_2 = np.squeeze(np.where(flag2 == 1))
        # print(len(string1))
        # print(len(string2))
        # print(minmax1)
        # print(minmax2)
        # z1_strings = torch.Tensor(string1).unsqueeze(0).to(device)
        # z2_strings = torch.Tensor(string2).unsqueeze(0).to(device)
        
        #
        # x_shape = torch.Tensor(x_shape).int().to(device)
        y_shape = x_shape//16
        z_shape = y_shape//4
        
        z1_hat = self.entropy_bottleneck1.decompress([string1], z_shape)  # z解码后结果
        z2_hat = self.entropy_bottleneck2.decompress([string2], z_shape)  # z解码后结果
        
        #gmm1
        gmm1 = self._h_s1(z1_hat) #三要素
        

        #===================decoding for y1 and y2
        # 逐像素计算概率分布、进行熵编码->后续考虑加入joint 进一步提高
        output2 = os.path.join(output_path,str(output_name)+ '.bin')
        
        # samples1 = np.arange(0, minmax1*2+1)
        # samples1 = torch.Tensor(samples1).to('cuda:0')
        samples1_np = np.arange(0, minmax1*2+1)
        samples2_np = np.arange(0, minmax2*2+1)

        start = time.time()
        #
        # y1_hat = np.zeros([1] + [y_shape[1]+kernel_size-1] + [y_shape[2]+kernel_size-1] + [num_filters])
        y1_hat = np.zeros([1]+[self.M]+[y_shape[0],y_shape[1]]) #[1,192,32,32]
        y2_hat = np.zeros([1]+[self.M]+[y_shape[0],y_shape[1]]) #[1,192,32,32]

        decoder = RangeDecoder(output2)
        #y1
        for i in range(len(non_zero_idx_1)):
            samples1 = torch.Tensor(samples1_np).to('cuda:0')
            samples1 = samples1.reshape([samples1.shape[0],1,1])
            samples1 = samples1.expand(samples1.shape[0],y_shape[0],y_shape[1]) #[arange(xx),width,height]
            #非零通道：一个通道算一次-> 并行优化
            ch_idx = non_zero_idx_1[i] #非零通道
            
            #gmm1 = [scales, means,weights] 通道数均为M*K
            ch_idx_list = [ch_idx+temp_i*self.M for temp_i in range(self.K)]
            sigma1 = gmm1[0][0,ch_idx_list,:,:]
            means1 = gmm1[1][0,ch_idx_list,:,:] + minmax1  #注意这里~~~ 相当于平移
            weights1 = gmm1[2][0,ch_idx_list,0,0] #(M*k):((k+1)*M) #1x1
            #torch.Size([5])
            # print(sigma1.shape)
            # print(means1.shape)
            # print(weights1.shape)
            
            #计算pmf
            pmf = None
            for temp_K in range(self.K):
                half = float(0.5)

                values = samples1 - means1[temp_K]

                temp_scales = self.gaussian1.lower_bound_scale(sigma1[temp_K])  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

                values = abs(values)
                upper = self._standardized_cumulative((half - values) / temp_scales)
                lower = self._standardized_cumulative((-half - values) / temp_scales)
                if pmf==None:
                    pmf = (upper - lower)* weights1[temp_K] #指定对应的M个通道
                else:
                    pmf += (upper - lower)* weights1[temp_K] #指定对应的M个通道

            # print(pmf.shape)
            # raise ValueError("stop")
            # print("minmax1",minmax1)
            
            pmf = pmf.cpu().numpy()
            # print(pmf.shape)
            # raise ValueError("stop1")        
            
            # 逐像素解码
            for h_idx in range(y_shape[0]):
                for w_idx in range(y_shape[1]):
                    pmf_temp = pmf[:,h_idx,w_idx]

                    # To avoid the zero-probability            
                    pmf_clip = np.clip(pmf_temp, 1.0/65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(i) for i in cdf]

                    y1_hat[0,ch_idx,h_idx,w_idx] = decoder.decode(1, cdf)[0] - minmax1 

        # print(y1_hat.shape)
        y1_hat = torch.Tensor(y1_hat).to(device)
        #end y1

        #decoding for x1 and gmm2
        x1_hat,g1_4,g1_5,g1_6 = self.decoder1(y1_hat)
        y1_global_context = self._global_context(y1_hat)  # 用y1_hat 看补充材料的图注说明  #返回tuple(3)
        gmm2 = self._h_s2(z2_hat, y1_hat)  # 三要素
        #y2
        for i in range(len(non_zero_idx_2)):
            samples2 = torch.Tensor(samples2_np).to('cuda:0')
            samples2 = samples2.reshape([samples2.shape[0],1,1])
            samples2 = samples2.expand(samples2.shape[0],y_shape[0],y_shape[1]) #[arange(xx),width,height]
            #非零通道：一个通道算一次-> 并行优化
            ch_idx = non_zero_idx_2[i] #非零通道
            
            #gmm2 = [scales, means,weights] 通道数均为M*K
            ch_idx_list = [ch_idx+temp_i*self.M for temp_i in range(self.K)]
            sigma2 = gmm2[0][0,ch_idx_list,:,:]
            means2 = gmm2[1][0,ch_idx_list,:,:] + minmax2  #注意这里~~~ 相当于平移
            weights2 = gmm2[2][0,ch_idx_list,0,0] #(M*k):((k+1)*M) #1x1
            #torch.Size([5])
            # print(sigma2.shape)
            # print(means2.shape)
            # print(weights2.shape)
            
            #计算pmf
            pmf = None
            for temp_K in range(self.K):
                half = float(0.5)

                values = samples2 - means2[temp_K]

                temp_scales = self.gaussian2.lower_bound_scale(sigma2[temp_K])  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

                values = abs(values)
                upper = self._standardized_cumulative((half - values) / temp_scales)
                lower = self._standardized_cumulative((-half - values) / temp_scales)
                if pmf==None:
                    pmf = (upper - lower)* weights2[temp_K] #指定对应的M个通道
                else:
                    pmf += (upper - lower)* weights2[temp_K] #指定对应的M个通道

            # print(pmf.shape)
            # raise ValueError("stop")
            # print("minmax2",minmax2)
            
            pmf = pmf.cpu().numpy()
            # print(pmf.shape)
            # raise ValueError("stop2")        
            
            # 逐像素解码
            for h_idx in range(y_shape[0]):
                for w_idx in range(y_shape[1]):
                    pmf_temp = pmf[:,h_idx,w_idx]

                    # To avoid the zero-probability            
                    pmf_clip = np.clip(pmf_temp, 1.0/65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(i) for i in cdf]

                    y2_hat[0,ch_idx,h_idx,w_idx] = decoder.decode(1, cdf)[0] - minmax2

        # print(y2_hat.shape)
        y2_hat = torch.Tensor(y2_hat).to(device)
        #end y2

        # decode for x2
        #pic2_1
        pic2_g_s_c1 = self.pic2_g_s_conv1(y2_hat)  # Tensor
        pic2_g_s_g1 = self.pic2_g_s_gdn1(pic2_g_s_c1)
        #end pic2_1

        #pic2_2
        y1_cost_4 = self._cost_volume4(g1_4, pic2_g_s_g1, y1_global_context[2])
        y1_warp_4 = self._warp4(g1_4, y1_cost_4)
        pic2_g_s_c2 = self.pic2_g_s_conv2(torch.cat((y1_warp_4, pic2_g_s_g1), dim=-3))  # Tensor
        pic2_g_s_g2 = self.pic2_g_s_gdn2(pic2_g_s_c2)
        #end pic2_2

        #pic2_3
        y1_cost_5 = self._cost_volume5(g1_5, pic2_g_s_g2, y1_global_context[1])
        y1_warp_5 = self._warp5(g1_5, y1_cost_5)
        pic2_g_s_c3 = self.pic2_g_s_conv3(torch.cat((y1_warp_5, pic2_g_s_g2), dim=-3))  # Tensor
        pic2_g_s_g3 = self.pic2_g_s_gdn3(pic2_g_s_c3)
        #end pic2_3

        #pic2_4
        y1_cost_6 = self._cost_volume6(g1_6, pic2_g_s_g3, y1_global_context[0])
        y1_warp_6 = self._warp6(g1_6, y1_cost_6)
        x2_hat = self.pic2_g_s_conv4(torch.cat((y1_warp_6, pic2_g_s_g3), dim=-3))  # Tensor #即为x2_hat = pic2_g_s_c4
        #end pic2_4
        # end x2
        ###
        # raise ValueError('stop')
        
        decoder.close()
        return {
            'x1_hat':x1_hat,
            'x2_hat':x2_hat,
            'y1_hat':y1_hat,
            'y2_hat':y2_hat,
            'z1_hat':z1_hat,
            'z2_hat':z2_hat,
        }

##合体版 HESIC
class DSIC_plus(nn.Module):
    def __init__(self,N=128,M=192,F=21,C=32,K=5,**kwargs):
        super().__init__()
        self.m1 = DSIC(N=N,M=M,F=F,C=C,K=K)
        self.m2 = Independent_EN()

    def forward(self,x1,x2):
        out1 = self.m1(x1,x2)

        out2 = self.m2(out1['x1_hat'],out1['x2_hat'])
        # out1['x1_hat'] = out2['x1_hat']
        # out1['x2_hat'] = out2['x2_hat']
        # return out1

        return {
            'x1_hat': out2['x1_hat'],
            'x2_hat': out2['x2_hat'],
            'likelihoods': out1['likelihoods'],
        }
