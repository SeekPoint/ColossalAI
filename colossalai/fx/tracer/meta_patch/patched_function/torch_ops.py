import torch
from pydebug import gd, infoTensor
from ...registry import meta_patched_function


@meta_patched_function.register(torch.arange)
def torch_arange(*args, **kwargs):
    gd.debuginfo(prj="mt", info=f'')

    n = len(args)
    step = 1
    if n == 1:
        start = 0
        end = args[0]
    elif n == 2:
        start, end = args
    else:
        start, end, step = args
    if isinstance(start, float):
        start = int(start)
    if isinstance(end, float):
        start = int(end)
    if isinstance(step, float):
        step = int(step)
    step = kwargs.get("step", step)
    dtype = kwargs.get("dtype")
    return torch.empty((end - start) // step, dtype=dtype, device="meta")


@meta_patched_function.register(torch.finfo)
def torch_finfo(*args):
    gd.debuginfo(prj="mt", info=f'')
    return torch.finfo(*args)


@meta_patched_function.register(torch.where)
def torch_where(condition, x, y):
    gd.debuginfo(prj="mt", info=f'')
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


@meta_patched_function.register(torch.Tensor.repeat)
def torch_tensor_repeat(self, *sizes):
    gd.debuginfo(prj="mt", info=f'')
    shape = list(self.shape)
    for i, x in enumerate(sizes):
        shape[i] *= x
    return torch.empty(shape, device="meta")


@meta_patched_function.register(torch.index_select)
def torch_index_select(input, dim, index, *, out=None):
    gd.debuginfo(prj="mt", info=f'')
    shape = list(input.shape)
    shape[dim] = len(index)
    return torch.empty(*shape, device="meta")


@meta_patched_function.register(torch.Tensor.index_select)
def torch_tensor_index_select(self, dim, index):
    gd.debuginfo(prj="mt", info=f'')
    return torch_index_select(self, dim, index)


@meta_patched_function.register(torch.squeeze)
def torch_squeeze(input, dim=None):
    gd.debuginfo(prj="mt", info=f'')
    shape = list(input.shape)
    if dim is not None:
        if dim < 0:
            dim = input.dim() + dim
            gd.debuginfo(prj="mt", info=f'')
        if shape[dim] == 1:
            shape.pop(dim)
            gd.debuginfo(prj="mt", info=f'')
    else:
        gd.debuginfo(prj="mt", info=f'')
        new_shape = []
        for dim_value in shape:
            if dim_value == 1:
                continue
            new_shape.append(dim_value)
        shape = new_shape
    return torch.empty(shape, device="meta")


@meta_patched_function.register(torch.Tensor.squeeze)
def torch_tensor_squeeze(self, dim=None):
    gd.debuginfo(prj="mt", info=f'')
    return torch_squeeze(self, dim)


@meta_patched_function.register(torch.unsqueeze)
def torch_unsqueeze(input, dim):
    gd.debuginfo(prj="mt", info=f'')
    shape = list(input.shape)
    if dim < 0:
        dim = input.dim() + 1 + dim
        gd.debuginfo(prj="mt", info=f'')
    shape.insert(dim, 1)
    return torch.empty(shape, device="meta")


@meta_patched_function.register(torch.Tensor.unsqueeze)
def torch_tensor_unsqueeze(self, dim):
    gd.debuginfo(prj="mt", info=f'')
    return torch_unsqueeze(self, dim)


@meta_patched_function.register(torch.cat)
def torch_cat(tensors, dim=None, axis=None, *, out=None):
    gd.debuginfo(prj="mt", info=f'')
    if dim is None and axis is None:
        dim = 0
        gd.debuginfo(prj="mt", info=f'')
    if dim is None and axis is not None:
        dim = axis
        gd.debuginfo(prj="mt", info=f'')
    if dim < 0:
        dim = tensors[0].dim() + dim
        gd.debuginfo(prj="mt", info=f'')
    shapes = [t.shape for t in tensors]
    shape = list(shapes[0])
    concatenated_dim = sum(shape[dim] for shape in shapes)
    final_shape = shape[:dim] + [concatenated_dim] + shape[dim + 1 :]
    return torch.empty(final_shape, device="meta")


@meta_patched_function.register(torch.repeat_interleave)
def torch_repeat_interleave(input, repeats, dim=None, output_size=None):
    assert isinstance(repeats, int) or isinstance(
        repeats, torch.Tensor
    ), "Argument 'repeats' should be of type 'torch.Tensor' or 'int'"
    gd.debuginfo(prj="mt", info=f'')
    shape = list(input.shape) if dim is not None else [input.numel()]
    dim = dim if dim is not None else 0
    dim = input.dim() + dim if dim < 0 else dim

    if isinstance(repeats, int):
        shape[dim] = shape[dim] * repeats
        gd.debuginfo(prj="mt", info=f'')
    elif isinstance(repeats, torch.Tensor):
        shape[dim] = repeats.sum()
        gd.debuginfo(prj="mt", info=f'')
    return torch.empty(shape, device="meta")


@meta_patched_function.register(torch.Tensor.repeat_interleave)
def torch_tensor_repeat_interleave(self, repeats, dim=None, *, output_size=None):
    gd.debuginfo(prj="mt", info=f'')
    return torch_repeat_interleave(self, repeats, dim, output_size)


@meta_patched_function.register(torch.roll)
def torch_roll(input, shifts, dims=None):
    gd.debuginfo(prj="mt", info=f'')
    return torch.empty(input.shape, device="meta")


@meta_patched_function.register(torch.full)
def torch_full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    gd.debuginfo(prj="mt", info=f'')
    assert out is None, "assigning result to out is not supported yet"
    return torch.empty(size, device="meta", dtype=dtype, layout=layout, requires_grad=requires_grad)


@meta_patched_function.register(torch.max)
def torch_max(input, dim=None, keepdim=False, *, out=None):
    assert out is None, "assigning value to out is not supported yet"
    if dim is not None:
        gd.debuginfo(prj="mt", info=f'')
        if isinstance(dim, int):
            gd.debuginfo(prj="mt", info=f'')
            shape = list(input.shape)
            shape.pop(dim)
            if keepdim:
                gd.debuginfo(prj="mt", info=f'')
                shape.insert(dim, 1)
            return torch.empty(shape, device="meta", dtype=input.dtype), torch.empty(
                shape, device="meta", dtype=input.dtype
            )
        elif isinstance(dim, torch.Tensor):
            gd.debuginfo(prj="mt", info=f'')
            # when dim is a 0D or 1D tensor, it will maintain the same shape
            num_dims = dim.dim()
            if num_dims in [0, 1]:
                gd.debuginfo(prj="mt", info=f'')
                return torch.empty_like(input, device="meta")
            else:
                raise ValueError(f"Expected dim to a 0D or 1D tensor but got {num_dims} dimensions")
    else:
        gd.debuginfo(prj="mt", info=f'')
        return torch.empty([], device="meta", dtype=input.dtype)


@meta_patched_function.register(torch.Tensor.cpu)
def torch_tensor_cpu(input):
    gd.debuginfo(prj="mt", info=f'')
    return input.clone()


@meta_patched_function.register(torch.Tensor.cuda)
def torch_tensor_cuda(input, *args, **kwargs):
    gd.debuginfo(prj="mt", info=f'')
    return input.clone()
