import warnings
from functools import partial
from typing import Callable, Dict, List
from pydebug import gd, infoTensor
import torch.nn as nn
from torch import Tensor, nn

from colossalai.shardformer.layer import FusedLayerNorm, Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D

from .._utils import getattr_
from ..modeling.jit import get_jit_fused_dropout_add_func
from ..modeling.opt import OPTPipelineForwards, get_jit_fused_opt_decoder_layer_forward, get_opt_flash_attention_forward
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "OPTPolicy",
    "OPTModelPolicy",
    "OPTForCausalLMPolicy",
    "OPTForSequenceClassificationPolicy",
    "OPTForQuestionAnsweringPolicy",
]


class OPTPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        gd.debuginfo(prj="mt", info=f'')
        if self.shard_config.enable_tensor_parallelism:
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size
            gd.debuginfo(prj="mt", info=f'')
            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)
                gd.debuginfo(prj="mt", info=f'')
        return self.model

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder, OPTDecoderLayer
        gd.debuginfo(prj="mt", info=f'')

        policy = {}
        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            gd.debuginfo(prj="mt", info=f"OPT dosen't support sequence parallelism now, will ignore the sequence parallelism flag.")

        if self.shard_config.enable_tensor_parallelism:
            gd.debuginfo(prj="mt", info=f'')
            policy[OPTDecoder] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="embed_tokens",
                        target_module=VocabParallelEmbedding1D,
                    )
                ]
            )
            policy[OPTDecoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="fc1",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="fc2",
                        target_module=Linear1D_Row,
                    ),
                ]
            )

            policy[OPTAttention] = ModulePolicyDescription(
                attribute_replacement={
                    "embed_dim": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                    "num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="q_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="k_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="v_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="out_proj",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            gd.debuginfo(prj="mt", info=f'')
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="final_layer_norm", target_module=FusedLayerNorm, ignore_if_not_exist=True
                ),
                policy=policy,
                target_key=OPTDecoder,
            )
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="self_attn_layer_norm", target_module=FusedLayerNorm, ignore_if_not_exist=True
                    ),
                    SubModuleReplacementDescription(
                        suffix="final_layer_norm", target_module=FusedLayerNorm, ignore_if_not_exist=True
                    ),
                ],
                policy=policy,
                target_key=OPTDecoderLayer,
            )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            gd.debuginfo(prj="mt", info=f'')
            self.append_or_create_method_replacement(
                description={
                    "forward": get_opt_flash_attention_forward(),
                },
                policy=policy,
                target_key=OPTAttention,
            )

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            gd.debuginfo(prj="mt", info=f'')
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_opt_decoder_layer_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=OPTDecoderLayer,
            )

        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "OPTModel":
            module = self.model.decoder
            gd.debuginfo(prj="mt", info=f'')
        else:
            module = self.model.model.decoder
            gd.debuginfo(prj="mt", info=f'')

        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.layers), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.embed_tokens)
            held_layers.append(module.embed_positions)
            held_layers.append(module.project_in)
            gd.debuginfo(prj="mt", info=f'')

        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.layers[start_idx:end_idx])

        if stage_manager.is_last_stage():
            held_layers.append(module.final_layer_norm)
            held_layers.append(module.project_out)
            gd.debuginfo(prj="mt", info=f'')
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        gd.debuginfo(prj="mt", info=f'')
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "OPTModel":
                module = self.model.decoder
                gd.debuginfo(prj="mt", info=f'')
            else:
                module = self.model.model.decoder
                gd.debuginfo(prj="mt", info=f'')

            layers_per_stage = Policy.distribute_layers(len(module.layers), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {"forward": partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )


class OPTModelPolicy(OPTPolicy):
    def __init__(self) -> None:
        gd.debuginfo(prj="mt", info=f'')
        super().__init__()

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTModel

        policy = super().module_policy()
        gd.debuginfo(prj="mt", info=f'')

        if self.pipeline_stage_manager:
            gd.debuginfo(prj="mt", info=f'')
            self.set_pipeline_forward(
                model_cls=OPTModel, new_forward=OPTPipelineForwards.opt_model_forward, policy=policy
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        gd.debuginfo(prj="mt", info=f'')
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in OPTModel."""
        return []


class OPTForCausalLMPolicy(OPTPolicy):
    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        policy = super().module_policy()
        gd.debuginfo(prj="mt", info=f'')

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="lm_head", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                ),
                policy=policy,
                target_key=OPTForCausalLM,
            )
            gd.debuginfo(prj="mt", info=f'')
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=OPTForCausalLM, new_forward=OPTPipelineForwards.opt_for_causal_lm_forward, policy=policy
            )
            gd.debuginfo(prj="mt", info=f'')

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        gd.debuginfo(prj="mt", info=f'')

        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
            gd.debuginfo(prj="mt", info=f'')
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        opt_model = self.model
        gd.debuginfo(prj="mt", info=f'')
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            num_stages = self.pipeline_stage_manager.num_stages
            gd.debuginfo(prj="mt", info=f'')
            if id(opt_model.model.decoder.embed_tokens.weight) == id(opt_model.lm_head.weight):
                gd.debuginfo(prj="mt", info=f'')
                return [{0: opt_model.model.decoder.embed_tokens.weight, num_stages - 1: opt_model.lm_head.weight}]
        return []

    def postprocess(self):
        if self.shard_config.enable_tensor_parallelism and self.pipeline_stage_manager is None:
            gd.debuginfo(prj="mt", info=f'')
            binding_map = {
                "model.decoder.embed_tokens": "lm_head",
            }

            for k, v in binding_map.items():
                src_mod = getattr_(self.model, k)
                dst_mod = getattr_(self.model, v)
                dst_mod.weight = src_mod.weight

        return self.model


class OPTForSequenceClassificationPolicy(OPTPolicy):
    def __init__(self) -> None:
        super().__init__()
        gd.debuginfo(prj="mt", info=f'')

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForSequenceClassification
        gd.debuginfo(prj="mt", info=f'')

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=OPTForSequenceClassification,
                new_forward=OPTPipelineForwards.opt_for_sequence_classification_forward,
                policy=policy,
            )
            gd.debuginfo(prj="mt", info=f'')

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        gd.debuginfo(prj="mt", info=f'')
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.score)
            gd.debuginfo(prj="mt", info=f'')
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        "no shared params in OPTForSequenceClassification"
        return []


class OPTForQuestionAnsweringPolicy(OPTPolicy):
    def __init__(self) -> None:
        super().__init__()
        gd.debuginfo(prj="mt", info=f'')

    def module_policy(self):
        from transformers.models.opt.modeling_opt import OPTForQuestionAnswering

        policy = super().module_policy()
        gd.debuginfo(prj="mt", info=f'')

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=OPTForQuestionAnswering,
                new_forward=OPTPipelineForwards.opt_for_question_answering_forward,
                policy=policy,
            )
            gd.debuginfo(prj="mt", info=f'')

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        gd.debuginfo(prj="mt", info=f'')

        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.qa_outputs)
            gd.debuginfo(prj="mt", info=f'')

        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        "no shared params in OPTForSequenceClassification"
        return []
