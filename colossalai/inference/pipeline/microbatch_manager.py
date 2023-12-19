from enum import Enum
from typing import Dict
from pydebug import gd, infoTensor
import torch

from ..tensor_parallel.batch_infer_state import BatchInferState
from ..tensor_parallel.kvcache_manager import MemoryManager

__all__ = "MicroBatchManager"


class Status(Enum):
    PREFILL = 1
    GENERATE = 2
    DONE = 3
    COOLDOWN = 4


class MicroBatchDescription:
    """
    This is the class to record the infomation of each microbatch, and also do some update operation.
    This clase is the base class of `HeadMicroBatchDescription` and `BodyMicroBatchDescription`, for more
    details, please refer to the doc of these two classes blow.

    Args:
        inputs_dict (Dict[str, torch.Tensor]): the inputs of current stage. The key should have `input_ids` and `attention_mask`.
        output_dict (Dict[str, torch.Tensor]): the outputs of previous stage. The key should have `hidden_states` and `past_key_values`.
    """

    def __init__(
        self,
        inputs_dict: Dict[str, torch.Tensor],
        max_input_len: int,
        max_output_len: int,
        cache_manager: MemoryManager,
    ) -> None:
        gd.debuginfo(prj="mt", info=f'')
        self.mb_length = inputs_dict["input_ids"].shape[-1]
        self.target_length = self.mb_length + max_output_len
        self.infer_state = BatchInferState.init_from_batch(
            batch=inputs_dict, max_input_len=max_input_len, max_output_len=max_output_len, cache_manager=cache_manager
        )
        # print(f"[init] {inputs_dict}, {max_input_len}, {max_output_len}, {cache_manager}, {self.infer_state}")

    def update(self, *args, **kwargs):
        pass

    @property
    def state(self):
        """
        Return the state of current micro batch, when current length is equal to target length,
        the state is DONE, otherwise GENERATE

        """
        # TODO: add the condition for early stopping
        if self.cur_length == self.target_length:
            gd.debuginfo(prj="mt", info=f'')
            return Status.DONE
        elif self.cur_length == self.target_length - 1:
            gd.debuginfo(prj="mt", info=f'')
            return Status.COOLDOWN
        else:
            gd.debuginfo(prj="mt", info=f'')
            return Status.GENERATE

    @property
    def cur_length(self):
        """
        Return the current sequnence length of micro batch

        """


class HeadMicroBatchDescription(MicroBatchDescription):
    """
    This class is used to record the infomation of the first stage of pipeline, the first stage should have attributes `input_ids` and `attention_mask`
    and `new_tokens`, and the `new_tokens` is the tokens generated by the first stage. Also due to the schdule of pipeline, the operation to update the
    information and the condition to determine the state is different from other stages.

    Args:
        inputs_dict (Dict[str, torch.Tensor]): the inputs of current stage. The key should have `input_ids` and `attention_mask`.
        output_dict (Dict[str, torch.Tensor]): the outputs of previous stage. The key should have `hidden_states` and `past_key_values`.

    """

    def __init__(
        self,
        inputs_dict: Dict[str, torch.Tensor],
        max_input_len: int,
        max_output_len: int,
        cache_manager: MemoryManager,
    ) -> None:
        gd.debuginfo(prj="mt", info=f'')
        super().__init__(inputs_dict, max_input_len, max_output_len, cache_manager)
        assert inputs_dict is not None
        assert inputs_dict.get("input_ids") is not None and inputs_dict.get("attention_mask") is not None
        self.input_ids = inputs_dict["input_ids"]
        self.attn_mask = inputs_dict["attention_mask"]
        self.new_tokens = None

    def update(self, new_token: torch.Tensor = None):
        if new_token is not None:
            self._update_newtokens(new_token)
            gd.debuginfo(prj="mt", info=f'')
        if self.state is not Status.DONE and new_token is not None:
            self._update_attnmask()
            gd.debuginfo(prj="mt", info=f'')

    def _update_newtokens(self, new_token: torch.Tensor):
        if self.new_tokens is None:
            self.new_tokens = new_token
            gd.debuginfo(prj="mt", info=f'')
        else:
            self.new_tokens = torch.cat([self.new_tokens, new_token], dim=-1)
            gd.debuginfo(prj="mt", info=f'')

    def _update_attnmask(self):
        self.attn_mask = torch.cat(
            (self.attn_mask, torch.ones((self.attn_mask.shape[0], 1), dtype=torch.int64, device="cuda")), dim=-1
        )

    @property
    def cur_length(self):
        """
        When there is no new_token, the length is mb_length, otherwise the sequence length is `mb_length` plus the length of new_token

        """
        if self.new_tokens is None:
            gd.debuginfo(prj="mt", info=f'')
            return self.mb_length
        else:
            gd.debuginfo(prj="mt", info=f'')
            return self.mb_length + len(self.new_tokens[0])


class BodyMicroBatchDescription(MicroBatchDescription):
    """
    This class is used to record the infomation of the stages except the first stage of pipeline, the stages should have attributes `hidden_states` and `past_key_values`,

    Args:
        inputs_dict (Dict[str, torch.Tensor]): will always be `None`. Other stages only receive hiddenstates from previous stage.
    """

    def __init__(
        self,
        inputs_dict: Dict[str, torch.Tensor],
        max_input_len: int,
        max_output_len: int,
        cache_manager: MemoryManager,
    ) -> None:
        gd.debuginfo(prj="mt", info=f'')
        super().__init__(inputs_dict, max_input_len, max_output_len, cache_manager)

    @property
    def cur_length(self):
        """
        When there is no kv_cache, the length is mb_length, otherwise the sequence length is `kv_cache[0][0].shape[-2]` plus 1

        """
        return self.infer_state.seq_len.max().item()


class MicroBatchManager:
    """
    MicroBatchManager is a class that manages the micro batch.

    Args:
        stage (int): stage id of current stage.
        micro_batch_size (int): the micro batch size.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.

    """

    def __init__(
        self,
        stage: int,
        micro_batch_size: int,
        micro_batch_buffer_size: int,
        max_input_len: int,
        max_output_len: int,
        cache_manager_list: MemoryManager,
    ):
        gd.debuginfo(prj="mt", info=f'')
        self.stage = stage
        self.micro_batch_size = micro_batch_size
        self.buffer_size = micro_batch_buffer_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.cache_manager_list = cache_manager_list
        self.mb_descrption_buffer = {}
        self.new_tokens_buffer = {}
        self.idx = 0

    def add_descrption(self, inputs_dict: Dict[str, torch.Tensor]):
        if self.stage == 0:
            gd.debuginfo(prj="mt", info=f'')
            self.mb_descrption_buffer[self.idx] = HeadMicroBatchDescription(
                inputs_dict, self.max_input_len, self.max_output_len, self.cache_manager_list[self.idx]
            )
        else:
            gd.debuginfo(prj="mt", info=f'')
            self.mb_descrption_buffer[self.idx] = BodyMicroBatchDescription(
                inputs_dict, self.max_input_len, self.max_output_len, self.cache_manager_list[self.idx]
            )

    def step(self, new_token: torch.Tensor = None):
        """
        Update the state if microbatch manager, 2 conditions.
        1. For first stage in PREFILL, receive inputs and outputs, `_add_descrption` will save its inputs.
        2. For other conditon, only receive the output of previous stage, and update the descrption.

        Args:
            inputs_dict (Dict[str, torch.Tensor]): the inputs of current stage. The key should have `input_ids` and `attention_mask`.
            output_dict (Dict[str, torch.Tensor]): the outputs of previous stage. The key should have `hidden_states` and `past_key_values`.
            new_token (torch.Tensor): the new token generated by current stage.
        """
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        # Add descrption first if the descrption is None
        self.cur_descrption.update(new_token)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        return self.cur_state

    def export_new_tokens(self):
        gd.debuginfo(prj="mt", info=f'')
        new_tokens_list = []
        for i in self.mb_descrption_buffer.values():
            new_tokens_list.extend(i.new_tokens.tolist())
        return new_tokens_list

    def is_micro_batch_done(self):
        gd.debuginfo(prj="mt", info=f'')
        if len(self.mb_descrption_buffer) == 0:
            return False
        for mb in self.mb_descrption_buffer.values():
            if mb.state != Status.DONE:
                return False
        return True

    def clear(self):
        gd.debuginfo(prj="mt", info=f'')
        self.mb_descrption_buffer.clear()
        for cache in self.cache_manager_list:
            cache.free_all()

    def next(self):
        gd.debuginfo(prj="mt", info=f'')
        self.idx = (self.idx + 1) % self.buffer_size

    def _remove_descrption(self):
        gd.debuginfo(prj="mt", info=f'')
        self.mb_descrption_buffer.pop(self.idx)

    @property
    def cur_descrption(self) -> MicroBatchDescription:
        gd.debuginfo(prj="mt", info=f'')
        return self.mb_descrption_buffer.get(self.idx)

    @property
    def cur_infer_state(self):
        if self.cur_descrption is None:
            gd.debuginfo(prj="mt", info=f'')
            return None
        gd.debuginfo(prj="mt", info=f'')
        return self.cur_descrption.infer_state

    @property
    def cur_state(self):
        """
        Return the state of current micro batch, when current descrption is None, the state is PREFILL

        """
        gd.debuginfo(prj="mt", info=f'')
        if self.cur_descrption is None:
            gd.debuginfo(prj="mt", info=f'')
            return Status.PREFILL
        return self.cur_descrption.state
