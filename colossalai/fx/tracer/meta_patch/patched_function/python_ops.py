import operator
from pydebug import gd, infoTensor
import torch

from colossalai.fx.proxy import ColoProxy

from ...registry import meta_patched_function


@meta_patched_function.register(operator.getitem)
def operator_getitem(a, b):
    # copied from huggingface.utils.fx
    def to_concrete(t):
        gd.debuginfo(prj="mt", info=f'')
        if isinstance(t, torch.Tensor):
            gd.debuginfo(prj="mt", info=f'')
            concrete = torch.ones_like(t, device="cpu")
            if concrete.dtype in [torch.float16, torch.float32, torch.float64, torch.int32]:
                gd.debuginfo(prj="mt", info=f'')
                concrete = concrete.to(torch.int64)
            return concrete
        return t

    def _slice_convert(slice_obj):
        gd.debuginfo(prj="mt", info=f'')
        attrs = {"start": slice_obj.start, "stop": slice_obj.stop, "step": slice_obj.step}
        new_attrs = _slice_attr_convert(attrs)
        attr_dict_to_tuple = (new_attrs["start"], new_attrs["stop"], new_attrs["step"])
        return slice(*attr_dict_to_tuple)

    def _slice_attr_convert(attrs):
        gd.debuginfo(prj="mt", info=f'')
        new_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, ColoProxy):
                new_attrs[key] = value.meta_data
                gd.debuginfo(prj="mt", info=f'')
            else:
                new_attrs[key] = value
                gd.debuginfo(prj="mt", info=f'')
        return new_attrs

    if isinstance(b, tuple):
        gd.debuginfo(prj="mt", info=f'')
        b = list(b)
        for index, element in enumerate(b):
            if isinstance(element, slice):
                b[index] = _slice_convert(element)
        b = tuple(b)
    elif isinstance(b, slice):
        b = _slice_convert(b)
        gd.debuginfo(prj="mt", info=f'')

    if isinstance(a, torch.Tensor):
        gd.debuginfo(prj="mt", info=f'')
        # TODO: infer shape without performing the computation.
        if isinstance(b, tuple):
            b = tuple(map(to_concrete, b))
            gd.debuginfo(prj="mt", info=f'')
        else:
            b = to_concrete(b)
            gd.debuginfo(prj="mt", info=f'')
        return operator.getitem(torch.empty_like(a, device="cpu"), b).to("meta")

    if isinstance(a, ColoProxy):
        gd.debuginfo(prj="mt", info=f'')
        # TODO: infer shape without performing the computation.
        if isinstance(b, tuple):
            b = tuple(map(to_concrete, b))
            gd.debuginfo(prj="mt", info=f'')
        else:
            b = to_concrete(b)
            gd.debuginfo(prj="mt", info=f'')
        return operator.getitem(torch.empty_like(a.meta_data, device="cpu"), b).to("meta")
    return operator.getitem(a, b)
