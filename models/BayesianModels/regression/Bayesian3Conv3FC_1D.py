import math
import torch
import torch.nn as nn
from layers.BBBConv1d import BBBConv1d
from layers.BBBLinear import BBBLinear
from layers.misc import FlattenLayer, ModuleWrapper
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchviz import make_dot

class BBB3Conv3FC_1D(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, init_log_noise):
        super(BBB3Conv3FC_1D, self).__init__()

        self.num_classes = outputs

        self.conv1 = BBBConv1d(inputs, 32, 5, alpha_shape=(1,1), padding=2, bias=False, name='conv1')
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = BBBConv1d(32, 64, 5, alpha_shape=(1,1), padding=2, bias=False, name='conv2')
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = BBBConv1d(64, 128, 5, alpha_shape=(1,1), padding=1, bias=False, name='conv3')
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        # out = [500, 128, 14]

        self.flatten = FlattenLayer(14 * 128)
        self.fc1 = BBBLinear(14 * 128, 1000, alpha_shape=(1,1), bias=False, name='fc1')
        self.soft5 = nn.Softplus()

        self.fc2 = BBBLinear(1000, 1000, alpha_shape=(1,1), bias=False, name='fc2')
        self.soft6 = nn.Softplus()

        self.fc3 = BBBLinear(1000, outputs, alpha_shape=(1,1), bias=False, name='fc3')

        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))

if __name__ == '__main__':
    model = BBB3Conv3FC_1D(6, 9)
    # dummy_input = Variable(torch.rand(500, 9, 128))
    # with SummaryWriter(comment='tcn') as w:
    #     w.add_grap
    x = torch.rand(500, 9, 128)
    g = make_dot(model(x), params=dict(list(model.named_parameters())+ [('x', x)]))
    g.view()