from typing import Any, List, Union

import torch
from pydebug import gd, infoTensor
from ..proxy import ColoProxy
from .meta_patch import meta_patched_function

__all__ = ["is_element_in_list", "extract_meta"]


def is_element_in_list(elements: Union[List[Any], Any], list_: List[Any]):
    if isinstance(elements, (tuple, list, set)):
        gd.debuginfo(prj="mt", info=f'')
        for ele in elements:
            if ele not in list_:
                return False, ele
    else:
        gd.debuginfo(prj="mt", info=f'')
        if elements not in list_:
            return False, elements

    return True, None


def extract_meta(*args, **kwargs):
    gd.debuginfo(prj="mt", info=f'')
    def _convert(val):
        if isinstance(val, ColoProxy):
            gd.debuginfo(prj="mt", info=f'')
            return val.meta_data
        elif isinstance(val, (list, tuple)):
            gd.debuginfo(prj="mt", info=f'')
            return type(val)([_convert(ele) for ele in val])

        return val

    new_args = [_convert(val) for val in args]
    new_kwargs = {k: _convert(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


def compute_meta_data_for_functions_proxy(target, args, kwargs):
    args_metas, kwargs_metas = extract_meta(*args, **kwargs)

    # fetch patched function
    if meta_patched_function.has(target):
        meta_target = meta_patched_function.get(target)
        gd.debuginfo(prj="mt", info=f'')
    elif meta_patched_function.has(target.__name__):
        meta_target = meta_patched_function.get(target.__name__)
        gd.debuginfo(prj="mt", info=f'')
    else:
        meta_target = target
        gd.debuginfo(prj="mt", info=f'')
    meta_out = meta_target(*args_metas, **kwargs_metas)
    if isinstance(meta_out, torch.Tensor):
        meta_out = meta_out.to(device="meta")
        gd.debuginfo(prj="mt", info=f'')

    return meta_out
