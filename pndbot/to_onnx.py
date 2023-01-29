import torch

from model import PndModel

pnd = PndModel()
pnd.load_state_dict(torch.load("../models/clstm-0.pt"))

pnd.eval()

x = torch.rand(64, 1000, 4)

with torch.no_grad():
    pnd.eval()

    torch.onnx.export(pnd,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "./models/onnx/pnd_clstm.onnx",   # where to save the model (can be a file or file-like object)
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})

    print(x.shape)