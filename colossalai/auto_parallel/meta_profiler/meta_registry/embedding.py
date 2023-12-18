from typing import List, Tuple

import torch
from pydebug import gd, infoTensor
from colossalai._analyzer._subclasses.flop_tensor import flop_mapping
from colossalai._analyzer.fx.node_util import compute_size_in_bytes
from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, OperationDataType, TrainCycleItem

from ..registry import meta_register

__all__ = ["embedding_meta_info"]


@meta_register.register(torch.nn.Embedding)
def embedding_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """torch.nn.Embedding metainfo generator

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """
    input_tensor = next(filter(lambda x: x.type == OperationDataType.ARG, args)).data
    weight_tensor = next(filter(lambda x: x.type == OperationDataType.PARAM, args)).data
    output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data
    gd.debuginfo(prj="mt", info=f'input_tensor={input_tensor}')
    gd.debuginfo(prj="mt", info=f'weight_tensor={weight_tensor}')
    gd.debuginfo(prj="mt", info=f'output_tensor={output_tensor}')

    # compute cost
    fwd_compute_cost = flop_mapping[torch.ops.aten.embedding.default]([weight_tensor, input_tensor], [output_tensor])
    bwd_compute_cost = flop_mapping[torch.ops.aten.embedding_dense_backward.default](
        [output_tensor, weight_tensor], [weight_tensor])

    gd.debuginfo(prj="mt", info=f'fwd_compute_cost={fwd_compute_cost}')
    gd.debuginfo(prj="mt", info=f'bwd_compute_cost={bwd_compute_cost}')

    compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)
    gd.debuginfo(prj="mt", info=f'compute_cost={compute_cost}')

    # memory cost
    # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
    # NOTE: during the backward phase of torch.nn.Embedding, it seems when the input is large enough, it will
    # have a temp memory which is kind of weird and we don't know the reason yet, so currently we just assume
    # that there will be no temp memory, as the temp memory is significantly smaller than the gradient memory
    fwd_memory_cost = MemoryCost(
        activation=compute_size_in_bytes([input_tensor, output_tensor]), parameter=0, temp=0, buffer=0
    )
    gd.debuginfo(prj="mt", info=f'fwd_memory_cost={fwd_memory_cost}')


    bwd_memory_cost = MemoryCost(activation=compute_size_in_bytes([weight_tensor]), parameter=0, temp=0, buffer=0)
    gd.debuginfo(prj="mt", info=f'bwd_memory_cost={bwd_memory_cost}')

    total_memory_cost = MemoryCost(activation=fwd_memory_cost.activation + bwd_memory_cost.activation)
    gd.debuginfo(prj="mt", info=f'total_memory_cost={total_memory_cost}')

    memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_memory_cost)
    gd.debuginfo(prj="mt", info=f'memory_cost={memory_cost}')

    # store fwd_in, fwd_buffer, fwd_out
    fwd_in = [torch.zeros_like(input_tensor)]
    fwd_buffer = []
    fwd_out = [torch.zeros_like(output_tensor)]

    return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out
