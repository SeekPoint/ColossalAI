#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from .base_grad_scaler import BaseGradScaler

__all__ = ["ConstantGradScaler"]
from pydebug import gd, infoTensor

class ConstantGradScaler(BaseGradScaler):
    """A gradient scaler which uses constant loss scale

    Args:
        initial_scale (float): the initial loss scale
        verbose (bool): whether to log messages
    """

    def __init__(self, initial_scale: int, verbose: bool):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        super().__init__(initial_scale, verbose)
        gd.debuginfo(prj="mt", info=f"Constant Gradient Scaler is initialized with scale {self.scale}")
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def update(self, overflow: bool) -> None:
        """Do nothing to keep the loss scale constant.

        Args:
            overflow (bool): whether overflow occurs
        """
