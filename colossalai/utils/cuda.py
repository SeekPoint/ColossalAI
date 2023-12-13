#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
import torch.distributed as dist
from pydebug import gd, infoTensor

def set_to_cuda(models):
    """Send model to gpu.

    :param models: nn.module or a list of module
    """
    if isinstance(models, list) and len(models) > 1:
        gd.debuginfo(prj="mt", info=f'')
        ret = []
        for model in models:
            ret.append(model.to(get_current_device()))
        return ret
    elif isinstance(models, list):
        gd.debuginfo(prj="mt", info=f'')
        return models[0].to(get_current_device())
    else:
        gd.debuginfo(prj="mt", info=f'')
        return models.to(get_current_device())


def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if torch.cuda.is_available():
        gd.debuginfo(prj="mt", info=f'')
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        gd.debuginfo(prj="mt", info=f'')
        return torch.device("cpu")


def synchronize():
    """Similar to cuda.synchronize().
    Waits for all kernels in all streams on a CUDA device to complete.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache():
    """Similar to cuda.empty_cache()
    Releases all unoccupied cached memory currently held by the caching allocator.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_device(index: Optional[int] = None) -> None:
    if index is None:
        index = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(index)
