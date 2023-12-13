import torch
from pydebug import gd, infoTensor
from ...registry import meta_patched_function


@meta_patched_function.register(torch.nn.functional.relu)
def torch_nn_func_relu(input, inplace=False):
    gd.debuginfo(prj="mt", info=f'')
    return torch.empty(input.shape, device="meta")
