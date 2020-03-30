import math
import torch
import torch.nn as nn
from layers.BBBConv1d import BBBConv1d
from layers.BBBLinear import BBBLinear
from layers.misc import FlattenLayer, ModuleWrapper
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchviz import make_dot

class BBB3Liner(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, init_log_noise):
        super(BBB3Liner, self).__init__()

        self.outputs = outputs

        self.fc1 = BBBLinear(inputs, 100, alpha_shape=(1,1), bias=False, name='fc1')
        self.soft5 = nn.Softplus()

        # self.fc2 = BBBLinear(100, 100, alpha_shape=(1,1), bias=False, name='fc2')
        # self.soft6 = nn.Softplus()

        self.fc3 = BBBLinear(100, outputs, alpha_shape=(1,1), bias=False, name='fc3')

        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))

if __name__ == '__main__':
    model = BBB3Liner(1, 4, 0)
    # dummy_input = Variable(torch.rand(500, 9, 128))
    # with SummaryWriter(comment='tcn') as w:
    #     w.add_grap
    x = torch.rand(500, 4)
    g = make_dot(model(x), params=dict(list(model.named_parameters())+ [('x', x)]))
    g.view()