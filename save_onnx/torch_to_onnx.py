import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torch.onnx
from bbb3conv3fc import BBB3Conv3FC_1D
import numpy as np
import onnx
import onnxruntime

# BBB3Conv3FC_1D to onnix
# 数据：UCI_HAR 输入数据形式 (9, 128) 类别 6
outputs = 6
inputs = 9
batch_size = 1

# Initialize model
torch_model = BBB3Conv3FC_1D(outputs, inputs)

# set the model to inference mode
torch_model.eval()

# Input to the model
x = torch.randn(batch_size, 9, 128, requires_grad=True)
torch_out, torch_kl  = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "bbb3conv3fc_1d.onnx",     # where to save the model (can be a file or file-like object)
                  verbose=True,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

onnx_model = onnx.load("bbb3conv3fc_1d.onnx")

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(onnx_model.graph)

ort_session = onnxruntime.InferenceSession("BBB3Conv3FC_1D.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0])
print("Exported model has been tested with ONNXRuntime, and the result looks good!")