from typing import Optional

from transformers import LlamaConfig, LlamaForCausalLM

from ..base import Actor
from pydebug import gd, infoTensor

class LlamaActor(Actor):
    """
    Llama Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (LlamaConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        config: Optional[LlamaConfig] = None,
        checkpoint: bool = False,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
    ) -> None:
        if pretrained is not None:
            model = LlamaForCausalLM.from_pretrained(pretrained)
            gd.debuginfo(prj="mt", info=f'')
        elif config is not None:
            model = LlamaForCausalLM(config)
            gd.debuginfo(prj="mt", info=f'')
        else:
            model = LlamaForCausalLM(LlamaConfig())
            gd.debuginfo(prj="mt", info=f'')

        if checkpoint:
            model.gradient_checkpointing_enable()
            gd.debuginfo(prj="mt", info=f'')

        super().__init__(model, lora_rank, lora_train_bias)
