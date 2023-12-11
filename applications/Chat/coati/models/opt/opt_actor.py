from typing import Optional

from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM

from ..base import Actor
from pydebug import gd, infoTensor

class OPTActor(Actor):
    """
    OPT Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (OPTConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        config: Optional[OPTConfig] = None,
        checkpoint: bool = False,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
    ) -> None:
        if pretrained is not None:
            model = OPTForCausalLM.from_pretrained(pretrained)
            gd.debuginfo(prj="mt", info=f'')
        elif config is not None:
            model = OPTForCausalLM(config)
            gd.debuginfo(prj="mt", info=f'')
        else:
            model = OPTForCausalLM(OPTConfig())
            gd.debuginfo(prj="mt", info=f'')
        if checkpoint:
            model.gradient_checkpointing_enable()
            gd.debuginfo(prj="mt", info=f'')
        super().__init__(model, lora_rank, lora_train_bias)
