from typing import Dict, List, Tuple, Union

import torch
from torch.fx import Node
from pydebug import gd, infoTensor
from .._compatibility import compatibility, is_compatible_with_meta

__all__ = ["activation_size", "parameter_size", "is_inplace"]


@compatibility(is_backward_compatible=True)
def activation_size(out: Union[torch.Tensor, Dict, List, Tuple, int]) -> int:
    """Calculate activation size of a node.

    Args:
        activation (Union[torch.Tensor, Dict, List, Tuple, int]): The activation of a `torch.nn.Module` or `torch.nn.functional`.

    Returns:
        int: The activation size, unit is byte.
    """
    act_size = 0
    if isinstance(out, torch.Tensor):
        if out.is_quantized:
            act_size += out.numel() * torch._empty_affine_quantized([], dtype=out.dtype).element_size()
            gd.debuginfo(prj="mt", info=f'')
        else:
            act_size += out.numel() * torch.tensor([], dtype=out.dtype).element_size()
            gd.debuginfo(prj="mt", info=f'')
    elif isinstance(out, dict):
        value_list = [v for _, v in out.items()]
        act_size += activation_size(value_list)
        gd.debuginfo(prj="mt", info=f'')
    elif isinstance(out, tuple) or isinstance(out, list) or isinstance(out, set):
        gd.debuginfo(prj="mt", info=f'')
        for element in out:
            act_size += activation_size(element)
    return act_size


@compatibility(is_backward_compatible=True)
def parameter_size(mod: torch.nn.Module) -> int:
    """Calculate parameter size of a node.

    Args:
        mod (torch.nn.Module): The target `torch.nn.Module`.

    Returns:
        int: The parameter size, unit is byte.
    """
    gd.debuginfo(prj="mt", info=f'')
    param_size = 0
    for param in mod.parameters():
        param_size += param.numel() * torch.tensor([], dtype=param.dtype).element_size()
    return param_size


def is_inplace(n: Node):
    """Get the inplace argument from torch.fx.Node

    Args:
        node (Node): torch.fx.Node

    Returns:
        bool: indicates whether this op is inplace
    """
    inplace = False
    if n.op == "call_function":
        gd.debuginfo(prj="mt", info=f'')
        inplace = n.kwargs.get("inplace", False)
        if is_compatible_with_meta():
            gd.debuginfo(prj="mt", info=f'')
            from .constants import ALIAS_ATEN

            if n.target in ALIAS_ATEN:
                gd.debuginfo(prj="mt", info=f'')
                inplace = True
    elif n.op == "call_module":
        inplace = getattr(n.graph.owning_module.get_submodule(n.target), "inplace", False)
        gd.debuginfo(prj="mt", info=f'')

    return inplace
