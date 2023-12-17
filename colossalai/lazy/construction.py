from contextlib import contextmanager
from typing import Callable, Dict, Tuple

import torch
from pydebug import gd, infoTensor
__all__ = [
    "_LEGACY_TENSOR_CONSTRUCTOR",
    "_NO_META_FACTORY",
    "_NORMAL_FACTORY",
    "ConstructorManager",
]

# reference: https://pytorch.org/cppdocs/notes/tensor_creation.html
_NORMAL_FACTORY = [
    "arange",
    "full",
    "empty",
    "linspace",
    "logspace",
    "ones",
    "rand",
    "randn",
    "randint",
    "randperm",
    "zeros",
    "tensor",
]

# factory function that does not support meta tensor backend
_NO_META_FACTORY = [
    "eye",
]

_LEGACY_TENSOR_CONSTRUCTOR = {
    "FloatTensor": torch.float,
    "DoubleTensor": torch.double,
    "HalfTensor": torch.half,
    "BFloat16Tensor": torch.bfloat16,
    "ByteTensor": torch.uint8,
    "CharTensor": torch.int8,
    "ShortTensor": torch.short,
    "IntTensor": torch.int,
    "LongTensor": torch.long,
    "BoolTensor": torch.bool,
}


class ConstructorManager:
    # function name: (new, old)
    overwrites: Dict[str, Tuple[Callable, Callable]] = {}
    changed: bool = False

    @staticmethod
    def apply(overwrites: Dict[Callable, Callable]):
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')
        ConstructorManager.overwrites.clear()
        gd.debuginfo(prj="mt", info=f'-1-')
        ConstructorManager.overwrites.update(overwrites)
        gd.debuginfo(prj="mt", info=f'-2-')
        ConstructorManager.redo()
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')

    @staticmethod
    def undo():
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')
        assert ConstructorManager.changed, "No constructor change to undo"
        for name, (new, old) in ConstructorManager.overwrites.items():
            setattr(torch, name, old)
        ConstructorManager.changed = False
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')

    @staticmethod
    def redo():
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')
        assert not ConstructorManager.changed, "Constructor already changed"
        for name, (new, old) in ConstructorManager.overwrites.items():
            setattr(torch, name, new)
        ConstructorManager.changed = True
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')

    @staticmethod
    @contextmanager
    def disable():
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')
        enabled = ConstructorManager.changed
        if enabled:
            gd.debuginfo(prj="mt", info=f'-----')
            ConstructorManager.undo()
        yield
        if enabled:
            gd.debuginfo(prj="mt", info=f'-----')
            ConstructorManager.redo()
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')

    @staticmethod
    def clear():
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')
        if ConstructorManager.changed:
            gd.debuginfo(prj="mt", info=f'-----')
            ConstructorManager.undo()
        gd.debuginfo(prj="mt", info=f'-----')
        ConstructorManager.overwrites.clear()
        gd.debuginfo(prj="mt", info=f'_FUNC_IN_OUT_')
