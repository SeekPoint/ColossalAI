import torch
from pydebug import gd, infoTensor
from ...registry import meta_patched_function


@meta_patched_function.register(torch.nn.functional.layer_norm)
def torch_nn_func_layernorm(input,
                            normalized_shape,
                            weight=None,
                            bias=None,
                            eps=1e-05):
    gd.debuginfo(prj="mt", info=f'')
    return torch.empty(input.shape, device="meta")


@meta_patched_function.register(torch.nn.functional.batch_norm)
def torch_nn_func_batchnorm(input,
                            running_mean,
                            running_var,
                            weight=None,
                            bias=None,
                            training=False,
                            momentum=0.1,
                            eps=1e-05):
    gd.debuginfo(prj="mt", info=f'')
    return torch.empty(input.shape, device="meta")
