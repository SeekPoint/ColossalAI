# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import torch
from pydebug import gd, infoTensor

class BaseOpHook(ABC):
    """This class allows users to add customized operations
    before and after the execution of a PyTorch submodule"""

    def __init__(self):
        pass

    @abstractmethod
    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    @abstractmethod
    def post_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    @abstractmethod
    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        pass

    @abstractmethod
    def post_bwd_exec(self, module: torch.nn.Module, input):
        pass

    @abstractmethod
    def post_iter(self):
        pass


# apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    gd.debuginfo(prj="mt", info=f'')
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module, functional, backward_function, output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        return functional.apply(module, backward_function, outputs)
    else:
        return outputs


class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        outputs = outputs.detach()
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        return outputs

    @staticmethod
    def backward(ctx, *args):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        ctx.pre_backward_function(ctx.module)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        ctx.module = module
        output = output.detach()
        ctx.pre_backward_function = pre_backward_function
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        return output

    @staticmethod
    def backward(ctx, *args):
        """
        Args:
            activation_grad of the next layer.
        Returns:
            grad of the input activation.
        """
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        ctx.pre_backward_function(ctx.module)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        return (None, None) + args


def register_ophooks_recursively(
    module: torch.nn.Module, ophook_list: List[BaseOpHook], name: str = "", filter_fn: Optional[Callable] = None
):
    gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
    r"""Recursively register pre/post hooks for all submodules in the module in FWD and BWD."""
    assert isinstance(module, torch.nn.Module)
    assert isinstance(ophook_list, (list, tuple))
    assert len(ophook_list) > 0, "expected at least 1 hook in the argument ophook_list but found 0"
    for hook in ophook_list:
        assert isinstance(hook, BaseOpHook)

    # Add hooks for submodules
    for child_name, child in module.named_children():
        register_ophooks_recursively(child, ophook_list, name + child_name, filter_fn)

    # Early return on modules with no parameters.
    if len(list(module.parameters(recurse=False))) == 0:
        gd.debuginfo(prj="mt", info=f'')
        return

    # return from filtered module
    if filter_fn is not None and filter_fn(module):
        gd.debuginfo(prj="mt", info=f'')
        return

    def _pre_forward_module_hook(submodule, *args):
        gd.debuginfo(prj="mt", info=f'')
        for hook in ophook_list:
            assert isinstance(submodule, torch.nn.Module)
            hook.pre_fwd_exec(submodule, *args)

    def _post_forward_module_hook(submodule, *args):
        gd.debuginfo(prj="mt", info=f'')
        for hook in ophook_list:
            assert isinstance(submodule, torch.nn.Module)
            hook.post_fwd_exec(submodule, *args)

    def _pre_backward_module_hook(submodule, inputs, output):
        gd.debuginfo(prj="mt", info=f'')
        def _run_before_backward_function(submodule):
            for hook in ophook_list:
                assert isinstance(submodule, torch.nn.Module)
                hook.pre_bwd_exec(submodule, inputs, output)

        return _apply_to_tensors_only(submodule, PreBackwardFunction, _run_before_backward_function, output)

    def _post_backward_module_hook(submodule, inputs):
        gd.debuginfo(prj="mt", info=f'')
        def _run_after_backward_function(submodule):
            for hook in ophook_list:
                assert isinstance(submodule, torch.nn.Module)
                hook.post_bwd_exec(submodule, inputs)

        return _apply_to_tensors_only(submodule, PostBackwardFunction, _run_after_backward_function, inputs)

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)

    gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
