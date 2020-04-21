import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet
from torch.autograd import Variable
import torch.onnx
import onnx

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)

if __name__ == "__main__":
    # onnx
    input_channels = 1
    n_classes = 10
    channel_sizes = [25, 25, 25, 25]
    kernel_size = 7
    dropout = 0.1
    dummy_input = Variable(torch.randn(10,1,784))
    model = TCN(input_channels, n_classes, channel_sizes, kernel_size, dropout)
    torch.onnx.export(model, dummy_input, "tcn.onnx", verbose=True)

    print("load model")
    # #load model
    # model_new = onnx.load("tcn.onnx")
    # # Check that the IR is well formed
    # onnx.checker.check_model(model_new)
    # # Print a human readable representation of the graph
    # onnx.helper.printable_graph(model_new.graph)

    #############
    # from onnx_tf.backend import prepare
    # import numpy as np
    # #############
    #
    # model = onnx.load("tcn.onnx")
    # tf_rep = prepare(model)
    #
    # img = np.load("./assets/image.npz")
    # output = tf_rep.run(img.reshape([1, 1, 28, 28]))

    from pytorch2keras import pytorch_to_keras

    import numpy as np
    input_np = np.random.uniform(0, 1, (10, 1, 784))
    input_var = Variable(torch.FloatTensor(input_np))


    k_model = pytorch_to_keras(model, input_var, [(784, 1)], verbose=True)

    # from pytorch2keras.converter import pytorch_to_keras
    #
    # # we should specify shape of the input tensor
    # k_model = pytorch_to_keras(model, input_var, [(None, None,)], verbose=True,change_ordering=True)

    # import onnx
    # from onnx2keras import onnx_to_keras
    #
    # # Load ONNX model
    # onnx_model = onnx.load('tcn.onnx')
    #
    # # Call the converter (input - is the main model input name, can be different for your model)
    # k_model = onnx_to_keras(onnx_model, ['input'],change_ordering=True)

    print('2')





