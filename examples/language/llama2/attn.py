from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from pydebug import gd, infoTensor

SUPPORT_XFORMERS = False
SUPPORT_FLASH2 = False
try:
    import xformers.ops as xops

    SUPPORT_XFORMERS = True
except ImportError:
    pass

try:
    from flash_attn import flash_attn_func

    SUPPORT_FLASH2 = True
except ImportError:
    pass

SUPPORT_FLASH = SUPPORT_XFORMERS or SUPPORT_FLASH2


def llama_flash_attention(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    bsz, q_len, _ = hidden_states.size()
    gd.debuginfo(prj="mt", info=f'bsz={(bsz)}, q_len={q_len}')

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    gd.debuginfo(prj="mt", info=f'query_states={infoTensor(query_states)}')
    gd.debuginfo(prj="mt", info=f'key_states={infoTensor(key_states)}')
    gd.debuginfo(prj="mt", info=f'value_states={infoTensor(value_states)}')

    kv_seq_len = key_states.shape[-2]
    gd.debuginfo(prj="mt", info=f'kv_seq_len={infoTensor(kv_seq_len)}')

    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
        gd.debuginfo(prj="mt", info=f'kv_seq_len={infoTensor(kv_seq_len)}')

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    gd.debuginfo(prj="mt", info=f'cos={infoTensor(cos)}')
    gd.debuginfo(prj="mt", info=f'sin={infoTensor(sin)}')

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    gd.debuginfo(prj="mt", info=f'query_states={infoTensor(query_states)}')
    gd.debuginfo(prj="mt", info=f'key_states={infoTensor(key_states)}')

    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
        gd.debuginfo(prj="mt", info=f'query_states={infoTensor(query_states)}')
        gd.debuginfo(prj="mt", info=f'key_states={infoTensor(key_states)}')

    past_key_value = (key_states, value_states) if use_cache else None
    gd.debuginfo(prj="mt", info=f'past_key_value={infoTensor(past_key_value)}')

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    gd.debuginfo(prj="mt", info=f'query_states={infoTensor(query_states)}')
    gd.debuginfo(prj="mt", info=f'key_states={infoTensor(key_states)}')

    # q, k, v is [B, H, S, K] and xformers need [B, S, H, K]. returns [B, S, H, K]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    gd.debuginfo(prj="mt", info=f'query_states={infoTensor(query_states)}')
    gd.debuginfo(prj="mt", info=f'key_states={infoTensor(key_states)}')
    gd.debuginfo(prj="mt", info=f'value_states={infoTensor(value_states)}')

    if SUPPORT_FLASH2:
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        gd.debuginfo(prj="mt", info=f'attn_output={infoTensor(attn_output)}')
    else:
        attn_output = xops.memory_efficient_attention(
            query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask()
        )
        gd.debuginfo(prj="mt", info=f'attn_output={infoTensor(attn_output)}')

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    gd.debuginfo(prj="mt", info=f'attn_output={infoTensor(attn_output)}')

    attn_output = self.o_proj(attn_output)
    gd.debuginfo(prj="mt", info=f'attn_output={infoTensor(attn_output)}')

    if not output_attentions:
        attn_weights = None

    gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    return attn_output, attn_weights, past_key_value


def replace_xformers(model: nn.Module):
    gd.debuginfo(prj="mt", info=f'')
    for module in model.modules():
        gd.debuginfo(prj="mt", info=f'module={module}')
        if isinstance(module, LlamaAttention):
            gd.debuginfo(prj="mt", info=f'')
            module.forward = MethodType(llama_flash_attention, module)
