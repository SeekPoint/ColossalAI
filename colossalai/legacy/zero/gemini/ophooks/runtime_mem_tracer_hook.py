from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List
from pydebug import gd, infoTensor
import torch

from colossalai.legacy.zero.gemini.tensor_utils import alloc_storage, free_storage
from colossalai.tensor.param_op_hook import ColoParamOpHook
from colossalai.zero.gemini.memory_tracer import MemStats, SyncCudaMemoryMonitor


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class GradMemStats:
    def __init__(self) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        self.unreleased_grad_flag = {}
        self.unreleased_grad_volume = 0
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def clear(self):
        self.unreleased_grad_flag.clear()
        self.unreleased_grad_volume = 0


class GradMemTracerHook:
    def __init__(self, grad_stats: GradMemStats):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        self.grad_hook_list = []
        self._grad_stats = grad_stats
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def grad_handle(self, p, grad):
        assert self._grad_stats.unreleased_grad_flag[p]
        free_storage(grad)
        self._grad_stats.unreleased_grad_volume -= grad.numel() * grad.element_size()
        self._grad_stats.unreleased_grad_flag[p] = False

    def register_grad_hook(self, module: torch.nn.Module):
        for p in module.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(partial(self.grad_handle, p)))
                self._grad_stats.unreleased_grad_flag[p] = False

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()


class ParamMemTracerHook(ColoParamOpHook):
    def __init__(self, memstats: MemStats, gradstats: GradMemStats) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        super().__init__()
        self._training_phase = TrainingPhase.FORWARD
        self._memstats = memstats
        self._grad_stats = gradstats
        self.mem_monitor = SyncCudaMemoryMonitor()
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def _free_cuda_params(self, params):
        gd.debuginfo(prj="mt", info=f'')
        for p in params:
            if p.data.device.type == "cpu":
                raise NotImplementedError("Only free cuda memory")
            free_storage(p.data)

    def _allocate_params_on_cuda(self, params: List[torch.nn.Parameter]):
        """
        move params to cuda

        Args:
            params (List[torch.nn.Parameter]): target params

        Raises:
            NotImplementedError: raise error when param has cpu grad
        """
        gd.debuginfo(prj="mt", info=f'')
        for p in params:
            cur_dev = p.data.device.type
            if cur_dev == "cpu":
                if p.grad is not None and p.grad.device.type == "cpu":
                    raise NotImplementedError("Only run in forward propagation")
                p.data = torch.empty(
                    p.data.shape, device="cuda", dtype=p.data.dtype, requires_grad=p.data.requires_grad
                )
            elif cur_dev == "cuda":
                alloc_storage(p.data)

    def record_model_data_volume(self, params):
        """
        get cuda model data used by params
        """
        gd.debuginfo(prj="mt", info=f'')
        data_volume = self._grad_stats.unreleased_grad_volume
        for p in params:
            cur_model_data_volume = p.data.numel() * p.data.element_size()
            data_volume += cur_model_data_volume
            if self._training_phase == TrainingPhase.BACKWARD and p.requires_grad:
                # add param.grad, actually param.grad is None in this time
                data_volume += cur_model_data_volume
                if not self._grad_stats.unreleased_grad_flag[p]:
                    self._grad_stats.unreleased_grad_volume += cur_model_data_volume
                    self._grad_stats.unreleased_grad_flag[p] = True
        # record max non model data used for this Op
        self._memstats.record_max_cuda_model_data(data_volume)

    def pre_op(self, params):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        max_cuda_used_pre_op = self.mem_monitor.finish()
        # record max cuda overall data for prev OP.
        self._memstats.record_max_cuda_overall_data(max_cuda_used_pre_op)
        # record max cuda non model data for prev OP.
        self._memstats.calc_max_cuda_non_model_data()

        self._allocate_params_on_cuda(params)
        # record max cuda  model data for current OP
        self.record_model_data_volume(params)

        self.mem_monitor.start()
        self._memstats.increase_preop_step(params)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def post_op(self, params):
        gd.debuginfo(prj="mt", info=f'')
        self._free_cuda_params(params)

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        gd.debuginfo(prj="mt", info=f'')
        self.pre_op(params)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        gd.debuginfo(prj="mt", info=f'')
        self.post_op(params)

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        gd.debuginfo(prj="mt", info=f'')
        self.pre_op(params)

    def post_backward(self, params: List[torch.Tensor]) -> None:
        gd.debuginfo(prj="mt", info=f'')
        self.post_op(params)

    @contextmanager
    def switch_training_phase(self, training_phase: TrainingPhase = TrainingPhase.BACKWARD):
        gd.debuginfo(prj="mt", info=f'')
        old_training_phase = self._training_phase
        try:
            self._training_phase = training_phase
            yield
        finally:
            self._training_phase = old_training_phase

    switch_to_backward = switch_training_phase
    switch_to_forward = partial(switch_to_backward, training_phase=TrainingPhase.FORWARD)
