from functools import partial
from typing import List
from pydebug import gd, infoTensor
import torch
from torch.nn import Module
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)

from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription

# import colossalai
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy

from ..modeling._utils import init_to_get_rotary
from ..modeling.llama import LlamaInferenceForwards

try:
    from colossalai.kernel.triton import rmsnorm_forward

    HAS_TRITON_RMSNORM = True
except:
    print("you should install triton from https://github.com/openai/triton")
    HAS_TRITON_RMSNORM = False


def get_triton_rmsnorm_forward():
    if HAS_TRITON_RMSNORM:
        gd.debuginfo(prj="mt", info=f'')
        def _triton_rmsnorm_forward(self: LlamaRMSNorm, hidden_states: torch.Tensor):
            gd.debuginfo(prj="mt", info=f'')
            return rmsnorm_forward(hidden_states, self.weight.data, self.variance_epsilon)

        return _triton_rmsnorm_forward
    else:
        gd.debuginfo(prj="mt", info=f'')
        return None


class LlamaModelInferPolicy(LlamaForCausalLMPolicy):
    def __init__(self) -> None:
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        super().__init__()
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def module_policy(self):
        gd.debuginfo(prj="mt", info=f'')
        policy = super().module_policy()
        gd.debuginfo(prj="mt", info=f'policy={policy}')

        if self.shard_config.inference_gptq:
            gd.debuginfo(prj="mt", info=f'')
            from colossalai.inference.quant.gptq.cai_gptq import ColCaiQuantLinear, RowCaiQuantLinear

            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            }
            policy[LlamaDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=RowCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=RowCaiQuantLinear,
                        kwargs={"split_num": 1},
                    ),
                ],
            )

        self.shard_config._infer()

        infer_forward = LlamaInferenceForwards.llama_model_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaModel)

        infer_forward = LlamaInferenceForwards.llama_decoder_layer_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=LlamaDecoderLayer
        )

        infer_forward = LlamaInferenceForwards.llama_flash_attn_kvcache_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=LlamaAttention
        )

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=LlamaForCausalLM, new_forward=LlamaInferenceForwards.llama_causal_lm_forward, policy=policy
            )
        infer_forward = None
        if HAS_TRITON_RMSNORM:
            infer_forward = get_triton_rmsnorm_forward()

        if infer_forward is not None:
            method_replacement = {"forward": partial(infer_forward)}
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=LlamaRMSNorm
            )

        gd.debuginfo(prj="mt", info=f'policy={policy}')
        return policy

    def postprocess(self):
        gd.debuginfo(prj="mt", info=f'')
        init_to_get_rotary(self.model.model)
        return self.model

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        gd.debuginfo(prj="mt", info=f'')
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_first_stage():
            gd.debuginfo(prj="mt", info=f'')
            held_layers.append(self.model.lm_head)
        return held_layers
