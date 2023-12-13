from typing import Dict, List
from pydebug import gd, infoTensor
from torch.fx import Node

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    MemoryCost,
    OperationData,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.device.device_mesh import DeviceMesh

from .strategy_generator import OutputStrategyGenerator

__all__ = ["OutputGenerator"]


class OutputGenerator(OutputStrategyGenerator):
    """
    OutputGenerator is a generic class to generate strategies for Output Node.
    """

    def __init__(
        self,
        operation_data_mapping: Dict[str, OperationData],
        device_mesh: DeviceMesh,
        predecessor_nodes: List[Node],
        output_option: str,
    ):
        gd.debuginfo(prj="mt", info=f'')
        super().__init__(operation_data_mapping, device_mesh, predecessor_nodes)
        self.output_option = output_option

    def validate(self) -> bool:
        gd.debuginfo(prj="mt", info=f'')
        return super().validate()

    def update_compute_cost(self, strategy: ShardingStrategy):
        gd.debuginfo(prj="mt", info=f'')
        compute_cost = TrainCycleItem(fwd=10, bwd=10, total=20)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy):
        """
        Compute the memory cost per device with this specific strategy.
        """
        gd.debuginfo(prj="mt", info=f'')
        fwd_mem_cost = MemoryCost(activation=0, parameter=0)

        bwd_mem_cost = MemoryCost(activation=0, parameter=0)

        # compute total cost
        total_mem_cost = MemoryCost(activation=0, parameter=0)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    def replica_strategy(self) -> List[ShardingStrategy]:
        """
        Generate replica strategy for output node.
        """
        gd.debuginfo(prj="mt", info=f'')
        dim_partition_dict_mapping = {}
        dim_partition_dict_for_output = []
        for index, _ in enumerate(self.predecessor_nodes):
            mapping_name = f"input_{index}"
            gd.debuginfo(prj="mt", info=f'mapping_name={mapping_name}')
            if isinstance(self.op_data[mapping_name].data, (tuple, list)):
                dim_partition_dict_for_input = [{} for _ in range(len(self.op_data[mapping_name].data))]
            else:
                dim_partition_dict_for_input = {}
            dim_partition_dict_mapping[mapping_name] = dim_partition_dict_for_input
            dim_partition_dict_for_output.append(dim_partition_dict_for_input)

        if len(dim_partition_dict_for_output) == 1:
            gd.debuginfo(prj="mt", info=f'')
            dim_partition_dict_for_output = dim_partition_dict_for_output[0]
        else:
            gd.debuginfo(prj="mt", info=f'')
            dim_partition_dict_for_output = tuple(dim_partition_dict_for_output)

        dim_partition_dict_mapping["output"] = dim_partition_dict_for_output

        communication_action_mapping = {}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = "Replica Output"

        strategy = self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )
        return strategy

    def distributed_strategy(self, mesh_list: List[List[int]] = None) -> List[ShardingStrategy]:
        """
        Generate distributed strategy for output node.
        """
        # TODO: need to take care of the case when the first element of output only need to be sharded.
        output_op_data = self.op_data["output"]
        if isinstance(output_op_data.data, tuple):
            gd.debuginfo(prj="mt", info=f'')
            length = len(output_op_data.data)
            dim_partition_dict_mapping = {
                "output": [{0: mesh_list}] * length,
            }
        else:
            gd.debuginfo(prj="mt", info=f'')
            dim_partition_dict_mapping = {
                "output": {0: mesh_list},
            }
        for index, _ in enumerate(self.predecessor_nodes):
            mapping_name = f"input_{index}"
            gd.debuginfo(prj="mt", info=f'mapping_name={mapping_name}')
            dim_partition_dict_mapping[mapping_name] = {0: mesh_list}

        communication_action_mapping = {}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = "Distributed Output"

        strategy = self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )
        return strategy

    def collate_strategies(self) -> List[ShardingStrategy]:
        gd.debuginfo(prj="mt", info=f'')
        strategy_list = []
        mesh_list = [0, 1]
        if self.output_option == "replicated":
            strategy_list.append(self.replica_strategy())
            gd.debuginfo(prj="mt", info=f'')
        elif self.output_option == "distributed":
            strategy_list.append(self.distributed_strategy(mesh_list))
            gd.debuginfo(prj="mt", info=f'')

        return strategy_list
