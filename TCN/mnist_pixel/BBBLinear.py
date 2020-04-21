import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from mnist_pixel.metrics import calculate_kl
from .misc import ModuleWrapper


class BBBLinear(ModuleWrapper):
    
    def __init__(self, in_features, out_features, alpha_shape=(1, 1), bias=True, name='BBBLinear'):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_shape = alpha_shape
        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.kl_value = calculate_kl
        self.name = name


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):

        mean = F.linear(x, self.W)
        if self.bias is not None:
            mean = mean + self.bias

        sigma = torch.exp(self.log_alpha) * self.W * self.W

        std = torch.sqrt(1e-16 + F.linear(x * x, sigma))
        # if self.training:
        epsilon = std.data.new(std.size()).normal_()
        # else:
        #     epsilon = 0.0
        # Local reparameterization trick
        out = mean + std * epsilon

        return out

    def kl_loss(self):
        return self.W.nelement() * self.kl_value(self.log_alpha) / self.log_alpha.nelement()

