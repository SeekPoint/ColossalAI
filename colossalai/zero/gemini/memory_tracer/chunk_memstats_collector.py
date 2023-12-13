from typing import Optional

from colossalai.utils import get_current_device
from colossalai.zero.gemini.chunk import ChunkManager

from .memory_stats import MemStats
from .memstats_collector import MemStatsCollector
from pydebug import gd, infoTensor

class ChunkMemStatsCollector(MemStatsCollector):
    def __init__(self, chunk_manager: ChunkManager, memstats: Optional[MemStats] = None) -> None:
        """

        Memory Statistic Collector for Chunks.

        Args:
            chunk_manager (ChunkManager): the chunk manager.
            memstats (Optional[MemStats], optional): memory statistics collected by RMT. Defaults to None.
        """
        super().__init__(memstats)
        self._chunk_manager = chunk_manager
        gd.debuginfo(prj="mt", info=f'')

    # override
    def record_model_data_volume(self) -> None:
        """
        record model data volume on cuda and cpu.
        """
        gd.debuginfo(prj="mt", info=f'')
        if self._start_flag and not self.use_outside_memstats:
            cuda_mem = self._chunk_manager.total_mem["cuda"]
            self._memstats.record_max_cuda_model_data(cuda_mem)
            gd.debuginfo(prj="mt", info=f'')

    @property
    def cuda_margin_mem(self) -> float:
        gd.debuginfo(prj="mt", info=f'')
        from colossalai.legacy.utils.memory import colo_device_memory_capacity

        return colo_device_memory_capacity(get_current_device()) - self._memstats.max_overall_cuda
