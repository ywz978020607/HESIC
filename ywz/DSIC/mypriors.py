# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
# import torch.nn.functional as F

from compressai.entropy_models import (EntropyBottleneck, GaussianConditional)
from compressai.layers import GDN, MaskedConv2d
from compressai.ans import BufferedRansEncoder, RansDecoder  # pylint: disable=E0611,E0401

from compressai.models.utils import update_registered_buffers, conv, deconv

__all__ = [
    'CompressionModel',
    'ScaleHyperprior',
]


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """
    def __init__(self, entropy_bottleneck_channels, init_weights=True):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(
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


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """
    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        self.g_a_conv1 = conv(3, N)
        self.g_a_gdn1 = GDN(N)
        self.g_a_conv2 = conv(N, N)
        self.g_a_gdn2 = GDN(N)
        self.g_a_conv3 = conv(N, N)
        self.g_a_gdn3 = GDN(N)
        self.g_a_conv4 = conv(N, M)

        self.g_s_conv1 = deconv(M, N)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.g_s_conv2 = deconv(N, N)
        self.g_s_gdn2 = GDN(N, inverse=True)
        self.g_s_conv3 = deconv(N, N)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.g_s_conv4 = deconv(N, 3)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

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

        self.z = self.h_a(torch.abs(self.y))
        self.z_hat, self.z_likelihoods = self.entropy_bottleneck(self.z)  #量化+定义速率失真损失
        self.scales_hat = self.h_s(self.z_hat) #sigma
        self.y_hat, self.y_likelihoods = self.gaussian_conditional(self.y, self.scales_hat)  #编码导出y_hat时，依然需要解z_hat然后产出y_hat


        # self.x_hat = self.g_s(self.y_hat)
        self.g_s_c1 = self.g_s_conv1(self.y_hat)  # Tensor
        self.g_s_g1 = self.g_s_gdn1(self.g_s_c1)
        self.g_s_c2 = self.g_s_conv2(self.g_s_g1)  # Tensor
        self.g_s_g2 = self.g_s_gdn2(self.g_s_c2)
        self.g_s_c3 = self.g_s_conv3(self.g_s_g2)  # Tensor
        self.g_s_g3 = self.g_s_gdn3(self.g_s_c3)
        self.g_s_c4 = self.g_s_conv4(self.g_s_g3)  # Tensor
        self.x_hat = self.g_s_c4

        return {
            'y':self.y,
            'z':self.z,
            'x_hat': self.x_hat,
            'y_hat': self.y_hat,
            'z_hat': self.z_hat,
            'scales_hat': self.scales_hat,
            'likelihoods': {
                'y': self.y_likelihoods,
                'z': self.z_likelihoods
            },
        }

    # def load_state_dict(self, state_dict):
    #     # Dynamically update the entropy bottleneck buffers related to the CDFs
    #     update_registered_buffers(self.entropy_bottleneck,
    #                               'entropy_bottleneck',
    #                               ['_quantized_cdf', '_offset', '_cdf_length'],
    #                               state_dict)
    #     update_registered_buffers(
    #         self.gaussian_conditional, 'gaussian_conditional',
    #         ['_quantized_cdf', '_offset', '_cdf_length', 'scale_table'],
    #         state_dict)
    #     super().load_state_dict(state_dict)
    #
    # @classmethod
    # def from_state_dict(cls, state_dict):
    #     """Return a new model instance from `state_dict`."""
    #     N = state_dict['g_a.0.weight'].size(0)
    #     M = state_dict['g_a.6.weight'].size(0)
    #     net = cls(N, M)
    #     net.load_state_dict(state_dict)
    #     return net
    #
    # def update(self, scale_table=None, force=False):
    #     if scale_table is None:
    #         scale_table = get_scale_table()
    #     self.gaussian_conditional.update_scale_table(scale_table, force=force)
    #     super().update(force=force)

