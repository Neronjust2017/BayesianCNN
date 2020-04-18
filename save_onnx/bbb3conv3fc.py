import math
import torch
import torch.nn as nn
from layers.misc import FlattenLayer, ModuleWrapper
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchviz import make_dot
import sys
import torch.nn.functional as F
from torch.nn import Parameter
import utils
import metrics
import config_bayesian as cfg

class BBB3Conv3FC_1D(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs):
        super(BBB3Conv3FC_1D, self).__init__()

        self.num_classes = outputs

        self.conv1 = BBBConv1d(inputs, 32, 5, alpha_shape=(1,1), padding=2, bias=False, name='conv1',std_size=[10, 32, 128])
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = BBBConv1d(32, 64, 5, alpha_shape=(1,1), padding=2, bias=False, name='conv2',std_size=[10, 64, 63])
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = BBBConv1d(64, 128, 5, alpha_shape=(1,1), padding=1, bias=False, name='conv3',std_size=[10, 128, 29])
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        # out = [500, 128, 14]

        self.flatten = FlattenLayer(14 * 128)
        self.fc1 = BBBLinear(14 * 128, 1000, alpha_shape=(1,1), bias=False, name='fc1', std_size=[10, 1000])
        self.soft5 = nn.Softplus()

        self.fc2 = BBBLinear(1000, 1000, alpha_shape=(1,1), bias=False, name='fc2', std_size=[10, 1000])
        self.soft6 = nn.Softplus()

        self.fc3 = BBBLinear(1000, outputs, alpha_shape=(1,1), bias=False, name='fc3',std_size=[10, 6])

class BBBConv1d(ModuleWrapper):

    def __init__(self, in_channels, out_channels, kernel_size, alpha_shape, stride=1,
                 padding=0, dilation=1, bias=True, name='BBBConv1d', std_size = None):
        super(BBBConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # 2D: (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha_shape = alpha_shape
        self.groups = 1
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.kernel_size))  # 2D: *self.kernel_size
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1))  # 2D: (1, out_channels, 1, 1)
        else:
            self.register_parameter('bias', None)
        self.out_bias = lambda input, kernel: F.conv1d(input, kernel, self.bias, self.stride, self.padding,
                                                       self.dilation, self.groups)
        self.out_nobias = lambda input, kernel: F.conv1d(input, kernel, None, self.stride, self.padding, self.dilation,
                                                         self.groups)
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.reset_parameters()
        self.name = name
        if cfg.record_mean_var:
            self.mean_var_path = cfg.mean_var_dir + f"{self.name}.txt"
        self.std_size = std_size

    def reset_parameters(self):
        n = self.in_channels
        # for k in self.kernel_size:
        #     n *= k
        n *= self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

    def forward(self, x):

        mean = self.out_bias(x, self.weight)

        sigma = torch.exp(self.log_alpha) * self.weight * self.weight

        std = torch.sqrt(1e-16 + self.out_nobias(x * x, sigma))
        # if self.training:

        #############################
        # epsilon = std.data.new(std.size()).normal_()
        #############################

        # means = torch.zeros(std.size())
        # epsilon = torch.normal(mean=means,std=1.0)
        # print("std.size:", std.size())
        epsilon = torch.randn(self.std_size)

        # else:
        #     epsilon = 0.0

        # Local reparameterization trick
        out = mean + std * epsilon

        if cfg.record_mean_var and cfg.record_now and self.training and self.name in cfg.record_layers:
            utils.save_array_to_file(mean.cpu().detach().numpy(), self.mean_var_path, "mean")
            utils.save_array_to_file(std.cpu().detach().numpy(), self.mean_var_path, "std")

        return out

    def kl_loss(self):
        return self.weight.nelement() / self.log_alpha.nelement() * metrics.calculate_kl(self.log_alpha)

class BBBLinear(ModuleWrapper):

    def __init__(self, in_features, out_features, alpha_shape=(1, 1), bias=True, name='BBBLinear',std_size=None):
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
        self.kl_value = metrics.calculate_kl
        self.name = name
        if cfg.record_mean_var:
            self.mean_var_path = cfg.mean_var_dir + f"{self.name}.txt"
        self.std_size = std_size

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

        #############################
        # epsilon = std.data.new(std.size()).normal_()
        #############################

        # means = torch.zeros(std.size())
        # epsilon = torch.normal(mean=means, std=1.0)
        # print("std.size:", std.size())

        epsilon = torch.randn(self.std_size)

        # else:
        #     epsilon = 0.0
        # Local reparameterization trick
        out = mean + std * epsilon

        if cfg.record_mean_var and cfg.record_now and self.training and self.name in cfg.record_layers:
            utils.save_array_to_file(mean.cpu().detach().numpy(), self.mean_var_path, "mean")
            utils.save_array_to_file(std.cpu().detach().numpy(), self.mean_var_path, "std")

        return out

    def kl_loss(self):
        return self.W.nelement() * self.kl_value(self.log_alpha) / self.log_alpha.nelement()




