import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torch.onnx
from bbb3conv3fc import BBB3Conv3FC_1D
import onnx

# BBB3Conv3FC_1D to onnix
# 数据：UCI_HAR 输入数据形式 (9, 128) 类别 6
outputs = 6
inputs = 9
model = BBB3Conv3FC_1D(outputs, inputs)
dummy_input = Variable(torch.randn(10,9,128))
out,kl = model(dummy_input)
torch.onnx.export(model, dummy_input, "BBB3Conv3FC_1D.onnx", verbose=True)

model_onnx = onnx.load("BBB3Conv3FC_1D.onnx")
# Check that the IR is well formed
onnx.checker.check_model(model_onnx)
# Print a human readable representation of the graph
onnx.helper.printable_graph(model_onnx.graph)

