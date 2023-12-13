from typing import Optional
from pydebug import gd, infoTensor
from torch.nn import Module

__all__ = [
    "get_pretrained_path",
    "set_pretrained_path",
]


def get_pretrained_path(model: Module) -> Optional[str]:
    gd.debuginfo(prj="mt", info=f'')
    return getattr(model, "_pretrained", None)


def set_pretrained_path(model: Module, path: str) -> None:
    gd.debuginfo(prj="mt", info=f'')
    setattr(model, "_pretrained", path)
