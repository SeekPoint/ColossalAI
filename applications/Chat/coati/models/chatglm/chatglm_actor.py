from typing import Optional

from ..base import Actor
from .configuration_chatglm import ChatGLMConfig
from .modeling_chatglm import ChatGLMForConditionalGeneration
from pydebug import gd, infoTensor

class ChatGLMActor(Actor):
    """
    ChatGLM Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (ChatGLMConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.

    do not support lora for now.
    """

    def __init__(
        self, pretrained: str = None, config: Optional[ChatGLMConfig] = None, checkpoint: bool = False
    ) -> None:
        if pretrained is not None:
            gd.debuginfo(prj="mt", info=f'')
            model = ChatGLMForConditionalGeneration.from_pretrained(pretrained)
        elif config is not None:
            gd.debuginfo(prj="mt", info=f'')
            model = ChatGLMForConditionalGeneration(config)
        else:
            gd.debuginfo(prj="mt", info=f'')
            model = ChatGLMForConditionalGeneration(ChatGLMConfig())
        if checkpoint:
            gd.debuginfo(prj="mt", info=f'')
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank=0, lora_train_bias="none")
