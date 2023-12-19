import torch.nn as nn
from pydebug import gd, infoTensor

class ModelWrapper(nn.Module):
    """
    A wrapper class to define the common interface used by booster.

    Args:
        module (nn.Module): The model to be wrapped.
    """

    def __init__(self, module: nn.Module) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        super().__init__()
        self.module = module
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def unwrap(self):
        """
        Unwrap the model to return the original model for checkpoint saving/loading.
        """
        gd.debuginfo(prj="mt", info=f'')
        if isinstance(self.module, ModelWrapper):
            gd.debuginfo(prj="mt", info=f'')
            return self.module.unwrap()
        return self.module

    def forward(self, *args, **kwargs):
        gd.debuginfo(prj="mt", info=f'')
        return self.module(*args, **kwargs)


class AMPModelMixin:
    """This mixin class defines the interface for AMP training."""

    def update_master_params(self):
        """
        Update the master parameters for AMP training.
        """
