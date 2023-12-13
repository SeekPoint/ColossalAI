import torch.distributed as dist
from torch.distributed import ProcessGroup
from pydebug import gd, infoTensor

class BaseStore:
    def __init__(self, torch_pg: ProcessGroup):
        gd.debuginfo(prj="mt", info=f'')
        self._world_size = dist.get_world_size(group=torch_pg)
        self._local_rank = dist.get_rank(group=torch_pg)

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank
