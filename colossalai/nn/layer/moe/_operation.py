from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

COL_MOE_KERNEL_FLAG = False

try:
    from colossalai._C import moe
except:
    moe = None


def build_moe_if_not_prebuilt():
    gd.debuginfo(prj="mt", info=f'')
    # load moe kernel during runtime if not pre-built
    global moe
    if moe is None:
        from colossalai.kernel.op_builder import MOEBuilder

        moe = MOEBuilder().load()
        gd.debuginfo(prj="mt", info=f'')


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, group: Optional[ProcessGroup] = None) -> Tensor:
        global moe
        gd.debuginfo(prj="mt", info=f'')
        if moe is None:
            gd.debuginfo(prj="mt", info=f'')
            from colossalai.kernel.op_builder import MOEBuilder

            moe = MOEBuilder().load()

        if ctx is not None:
            gd.debuginfo(prj="mt", info=f'')
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            gd.debuginfo(prj="mt", info=f'')
            return inputs.unsqueeze(0)

        buffer_shape = (comm_size,) + inputs.shape
        outputs = torch.empty(buffer_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(outputs, comm_size, dim=0))
        dist.all_gather(buffer_list, inputs, group=group)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Tensor) -> Tuple[Tensor, None]:
        gd.debuginfo(prj="mt", info=f'')
        return ReduceScatter.forward(None, grad_outputs, ctx.comm_grp), None


class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, group: Optional[ProcessGroup] = None) -> Tensor:
        if ctx is not None:
            ctx.comm_grp = group
            gd.debuginfo(prj="mt", info=f'')

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            gd.debuginfo(prj="mt", info=f'')
            return inputs.squeeze(0)

        if not inputs.is_contiguous():
            gd.debuginfo(prj="mt", info=f'')
            inputs = inputs.contiguous()

        output_shape = inputs.shape[1:]
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(inputs, comm_size, dim=0))
        dist.reduce_scatter(outputs, buffer_list, group=group)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Tensor) -> Tuple[Tensor, None]:
        gd.debuginfo(prj="mt", info=f'')
        return AllGather.forward(None, grad_outputs, ctx.comm_grp), None


class AllToAll(torch.autograd.Function):
    """Dispatches input tensor [e, c, h] to all experts by all_to_all_single
    operation in torch.distributed.
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, group: Optional[ProcessGroup] = None) -> Tensor:
        gd.debuginfo(prj="mt", info=f'')
        if ctx is not None:
            ctx.comm_grp = group
            gd.debuginfo(prj="mt", info=f'')
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
            gd.debuginfo(prj="mt", info=f'')
        if dist.get_world_size(group) == 1:
            gd.debuginfo(prj="mt", info=f'')
            return inputs
        output = torch.empty_like(inputs)
        dist.all_to_all_single(output, inputs, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        gd.debuginfo(prj="mt", info=f'')
        return AllToAll.forward(None, *grad_outputs, ctx.comm_grp), None


class MoeDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, mask, dest_idx, ec):
        s = tokens.size(0)
        h = tokens.size(1)

        # load moe kernel during runtime if not pre-built
        build_moe_if_not_prebuilt()

        expert_input = moe.dispatch_forward(s, ec, h, tokens, mask, dest_idx)

        ctx.save_for_backward(mask, dest_idx)
        ctx.s = s
        ctx.h = h
        ctx.ec = ec
        gd.debuginfo(prj="mt", info=f'')

        return expert_input

    @staticmethod
    def backward(ctx, output_grad):
        mask, dest_idx = ctx.saved_tensors
        d_tokens = moe.dispatch_backward(ctx.s, ctx.ec, ctx.h, output_grad, mask, dest_idx)
        gd.debuginfo(prj="mt", info=f'')
        return d_tokens, None, None, None


class MoeCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_tokens, logits, mask, dest_idx, ec):
        gd.debuginfo(prj="mt", info=f'')
        assert logits.dtype == torch.float32

        s = logits.size(0)
        e = logits.size(1)
        c = ec // e
        h = expert_tokens.size(-1)

        # load moe kernel during runtime if not pre-built
        build_moe_if_not_prebuilt()

        fp16_flag = expert_tokens.dtype == torch.float16
        cb_input = expert_tokens.to(torch.float32) if fp16_flag else expert_tokens
        ctokens = moe.combine_forward(s, e, c, h, cb_input, logits, mask, dest_idx)
        output = ctokens.to(torch.float16) if fp16_flag else ctokens

        ctx.save_for_backward(expert_tokens, logits, mask, dest_idx)
        ctx.s = s
        ctx.e = e
        ctx.c = c
        ctx.h = h
        ctx.fp16_flag = fp16_flag

        gd.debuginfo(prj="mt", info=f'')

        return output

    @staticmethod
    def backward(ctx, tokens_grad):
        expert_tokens, logits, mask, dest_idx = ctx.saved_tensors

        cb_grad = tokens_grad.to(torch.float32) if tokens_grad.dtype is torch.float16 else tokens_grad
        cb_input = expert_tokens.to(torch.float32) if ctx.fp16_flag else expert_tokens
        d_expert, d_logits = moe.combine_backward(ctx.s, ctx.e, ctx.c, ctx.h, cb_grad, cb_input, logits, mask, dest_idx)
        d_expert = d_expert.to(torch.float16) if ctx.fp16_flag else d_expert

        gd.debuginfo(prj="mt", info=f'')

        return d_expert, d_logits, None, None, None


def moe_cumsum(inputs: Tensor):
    dim0 = inputs.size(0)
    flag = (dim0 <= 1024) or (dim0 <= 2048 and dim0 % 2 == 0) or (dim0 % 4 == 0)
    if flag and COL_MOE_KERNEL_FLAG:
        gd.debuginfo(prj="mt", info=f'')
        # load moe kernel during runtime if not pre-built
        build_moe_if_not_prebuilt()
        return moe.cumsum_sub_one(inputs)
    else:
        gd.debuginfo(prj="mt", info=f'')
        return torch.cumsum(inputs, dim=0) - 1
