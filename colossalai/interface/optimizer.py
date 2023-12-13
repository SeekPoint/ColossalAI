from typing import Union
from pydebug import gd, infoTensor
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


class OptimizerWrapper:
    """
    A standard interface for optimizers wrapped by the Booster.

    Args:
        optim (Optimizer): The optimizer to be wrapped.
    """

    def __init__(self, optim: Optimizer):
        gd.debuginfo(prj="mt", info=f'')
        self.optim = optim

    @property
    def parameters(self):
        gd.debuginfo(prj="mt", info=f'')
        params = []

        for group in self.param_groups:
            params += group["params"]
        return params

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        gd.debuginfo(prj="mt", info=f'')
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        gd.debuginfo(prj="mt", info=f'')
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        Performs a single optimization step.
        """
        gd.debuginfo(prj="mt", info=f'')
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """
        Clears the gradients of all optimized `torch.Tensor`.
        """
        gd.debuginfo(prj="mt", info=f'')
        self.optim.zero_grad(*args, **kwargs)

    def backward(self, loss: Tensor, *args, **kwargs):
        """
        Performs a backward pass on the loss.
        """
        gd.debuginfo(prj="mt", info=f'')
        loss.backward(*args, **kwargs)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor):
        gd.debuginfo(prj="mt", info=f'')
        torch.autograd.backward(tensor, grad)

    def state_dict(self):
        """
        Returns the optimizer state.
        """
        gd.debuginfo(prj="mt", info=f'')
        return self.optim.state_dict()

    def load_state_dict(self, *args, **kwargs):
        """
        Loads the optimizer state.
        """
        gd.debuginfo(prj="mt", info=f'')
        self.optim.load_state_dict(*args, **kwargs)

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        """
        Clips gradient of an iterable of parameters at specified min and max values.

        Args:
            clip_value (float or int): maximum allowed value of the gradients. Gradients are clipped in the range

        Note:
            In PyTorch Torch 2.0 and above, you can pass in foreach=True as kwargs to clip_grad_value_ to use the
            faster implementation. Please refer to the PyTorch documentation for more details.
        """
        gd.debuginfo(prj="mt", info=f'')
        nn.utils.clip_grad_value_(self.parameters, clip_value, *args, **kwargs)

    def clip_grad_by_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = False,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Clips gradient norm of an iterable of parameters.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
            error_if_nonfinite (bool): if True, an error is raised if the total norm is non-finite. Default: False

        Note:
            In PyTorch Torch 2.0 and above, you can pass in foreach=True as kwargs to clip_grad_norm_ to use the
            faster implementation. Please refer to the PyTorch documentation for more details.
        """
        gd.debuginfo(prj="mt", info=f'')
        norm = nn.utils.clip_grad_norm_(self.parameters, max_norm, norm_type, error_if_nonfinite, *args, **kwargs)
        return norm

    def scale_loss(self, loss: Tensor):
        """
        Scales the loss for mixed precision training.

        Note: Only available for optimizers with mixed precision training.

        Args:
            loss (Tensor): The loss to be scaled.
        """
        raise NotImplementedError(
            "The method scale_loss is only available for optimizers with mixed precision training"
        )

    def unscale_grad(self):
        """
        Unscale the gradients for mixed precision training.

        Note: Only available for optimizers with mixed precision training.
        """
        raise NotImplementedError(
            "The method unscale_grad is only available for optimizers with mixed precision training"
        )

    def unwrap(self):
        """
        Unwrap the optimizer for checkpoint saving/loading.
        """
        gd.debuginfo(prj="mt", info=f'')
        return self.optim
