import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn as nn
import math
from mnist_pixel.BBBConv1d import BBBConv1d
from mnist_pixel.BBBLinear import BBBLinear

class BBBTCN(nn.Module):
    def __init__(self, output_size, input_size, num_channels, kernel_size=7, dropout=0.05):
        super(BBBTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = BBBLinear(num_channels[-1], output_size)

        print("modules")
        for i, module in enumerate(self.modules()):
            if hasattr(module, 'kl_loss'):
                print(i, module)

        print("children")

        for i, module in enumerate(self.children()):
            if hasattr(module, 'kl_loss'):
                print(i, module)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])

        kl = 0.0
        for module in self.modules(): # self.modules()采用深度优先遍历的方式，存储了net的所有模块
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        return o,kl

class TemporalConvNet(nn.Module):
    def __init__(self,  num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(BBBConv1d(n_inputs, n_outputs, kernel_size,alpha_shape=(1,1),
                                           stride=stride, padding=padding, dilation=dilation, bias=False))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(BBBConv1d(n_outputs, n_outputs, kernel_size,alpha_shape=(1,1),
                                           stride=stride, padding=padding, dilation=dilation, bias=False))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = BBBConv1d(n_inputs, n_outputs, 1, alpha_shape=(1,1),bias=False) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        # self.init_weights()

    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0, 0.01)
    #     self.conv2.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()




