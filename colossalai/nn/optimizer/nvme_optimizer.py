import math
import os
import tempfile
from typing import Callable, Dict, List, Optional
from pydebug import gd, infoTensor
import torch
from torch.nn.parameter import Parameter


class NVMeOptimizer(torch.optim.Optimizer):
    """A base class for offloading optimizer states.

    Args:
        params: parameters
        defaults (dict): default dict
        nvme_offload_fraction (float, optional): Fraction of params to be offloaded to NVMe. Defaults to 0.0.
        offload_dir (Optional[str], optional): Directory to save NVMe offload files.
            If it's ``None``, a random temporary directory will be used. Defaults to None.

    Raises:
        ImportError: Raise if ``tensornvme`` is not installed.
    """

    def __init__(self,
                 params,
                 defaults: dict,
                 nvme_offload_fraction: float = 0.0,
                 offload_dir: Optional[str] = None) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        assert 0.0 <= nvme_offload_fraction <= 1.0
        super().__init__(params, defaults)
        self.nvme_offload_fraction = float(nvme_offload_fraction)
        if self.nvme_offload_fraction > 0.0:
            try:
                from tensornvme import DiskOffloader
                from tensornvme._C import get_backends
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Please install tensornvme to use NVMeOptimizer")
            self.offload_dir = offload_dir or tempfile.mkdtemp()
            backend = "uring" if "uring" in get_backends() else "aio"
            self.offloader = DiskOffloader(self.offload_dir, 8, backend=backend)
            gd.debuginfo(prj="mt", info=f'')
        else:
            self.offload_dir = None
            self.offloader = None
            gd.debuginfo(prj="mt", info=f'')

        self.is_on_nvme: Dict[Parameter, bool] = {}
        self.offloaded_numel: int = 0
        # As param may be not materialized here, these attributes are initialized when the first step
        self.total_numel: Optional[int] = None
        self.can_offload_numel: Optional[int] = None

        self.prefetch_params: List[Parameter] = []
        self.param_to_prefetch_idx: Dict[Parameter, int] = {}
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def _get_numel(self) -> int:
        gd.debuginfo(prj="mt", info=f'')
        numel = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel += p.storage().size()
        return numel

    def _post_state_init(self, param: Parameter) -> None:
        numel = param.storage().size()
        if (
            self.offloader is not None
            and param.device.type == "cpu"
            and numel + self.offloaded_numel <= self.can_offload_numel
        ):
            self.is_on_nvme[param] = True
            self.offloaded_numel += numel
            gd.debuginfo(prj="mt", info=f'param={infoTensor(param)}')
        else:
            self.is_on_nvme[param] = False
            gd.debuginfo(prj="mt", info=f'param={infoTensor(param)}')

    def _setup_prefetch_params(self) -> List[Parameter]:
        gd.debuginfo(prj="mt", info=f'')
        if self.offloader is None:
            gd.debuginfo(prj="mt", info=f'')
            return
        assert len(self.prefetch_params) == 0 and len(self.param_to_prefetch_idx) == 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if len(self.state[p]) > 0 and self.is_on_nvme[p]:
                    assert p.device.type == "cpu"
                    self.param_to_prefetch_idx[p] = len(self.prefetch_params)
                    self.prefetch_params.append(p)

    def _pre_step(self, *state_keys: str) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if self.total_numel is None:
            self.total_numel = self._get_numel()
            self.can_offload_numel = math.floor(self.total_numel * self.nvme_offload_fraction)
            gd.debuginfo(prj="mt", info=f'')

        self._setup_prefetch_params()
        if self.offloader is None or len(self.prefetch_params) == 0:
            gd.debuginfo(prj="mt", info=f'')
            return
        state = self.state[self.prefetch_params[0]]
        for key in state_keys:
            self.offloader.async_read(state[key])

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def _pre_update(self, param: Parameter, *state_keys: str) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if self.offloader is None or param not in self.param_to_prefetch_idx:
            gd.debuginfo(prj="mt", info=f'')
            return
        self.offloader.sync_read_events()
        idx = self.param_to_prefetch_idx[param]
        if idx + 1 < len(self.prefetch_params):
            state = self.state[self.prefetch_params[idx + 1]]
            for key in state_keys:
                self.offloader.async_read(state[key])
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def _post_update(self, param: Parameter, *state_keys: str) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if self.offloader is None:
            gd.debuginfo(prj="mt", info=f'')
            return
        self.offloader.sync_write_events()
        if self.is_on_nvme[param]:
            gd.debuginfo(prj="mt", info=f'')
            state = self.state[param]
            for key in state_keys:
                self.offloader.async_write(state[key])
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def _post_step(self) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if self.offloader is not None:
            gd.debuginfo(prj="mt", info=f'')
            self.offloader.synchronize()
            self.prefetch_params.clear()
            self.param_to_prefetch_idx.clear()
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        """Performs a single optimization step (parameter update).

        Example:

            >>> self._pre_step('exp_avg', 'exp_avg_sq')
            >>> for group in self.param_groups:
            >>>     for p in group['params']:
            >>>         if p.grad is None:
            >>>             continue
            >>>         state = self.state[p]
            >>>         if len(state) == 0:
            >>>             state['exp_avg'] = ...
            >>>             state['exp_avg_sq'] = ...
            >>>             self._post_state_init(p)
            >>>         if p.device.type == 'cpu':
            >>>             self._pre_update(p, 'exp_avg', 'exp_avg_sq')
            >>>             adam()
            >>>             self._post_update(p, 'exp_avg', 'exp_avg_sq')
            >>>         else:
            >>>             ...
            >>> self._post_step()

        Args:
            closure (Optional[Callable[[], float]], optional): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError

    def state_dict(self) -> dict:
        gd.debuginfo(prj="mt", info=f'')
        # TODO(ver217): design a new method to save state_dict. When using NVMe offload, this method may lead to OOM.
        if self.offloader is not None:
            raise NotImplementedError
        return super().state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        gd.debuginfo(prj="mt", info=f'')
        # TODO(ver217): design a new method to load state_dict. When using NVMe offload, whole state_dict may not be able to fit in memory.
        if self.offloader is not None:
            raise NotImplementedError
        super().load_state_dict(state_dict)

    def __del__(self) -> None:
        gd.debuginfo(prj="mt", info=f'self.offload_dir={self.offload_dir}')
        if getattr(self, "offloader", None) is not None:
            del self.offloader
            if os.path.exists(self.offload_dir):
                try:
                    os.rmdir(self.offload_dir)
                except OSError as e:
                    gd.debuginfo(prj="mt", info=f'expception={e}')
