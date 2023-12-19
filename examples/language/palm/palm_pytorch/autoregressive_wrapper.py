import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# helper function
from pydebug import gd, infoTensor

def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# top k filtering


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    gd.debuginfo(prj="mt", info=f'k={k}, val={val}, ind={ind}')

    probs = torch.full_like(logits, float("-inf"))
    gd.debuginfo(prj="mt", info=f'probs={infoTensor(probs)}')

    probs.scatter_(1, ind, val)
    gd.debuginfo(prj="mt", info=f'probs={infoTensor(probs)}')
    return probs


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, max_seq_len=2048, pad_value=0):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.net = net
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    @torch.no_grad()
    @eval_decorator
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0, filter_thres=0.9, **kwargs):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        b, t, device = *start_tokens.shape, start_tokens.device

        out = start_tokens

        for _ in range(seq_len):
            logits = self.net(out, **kwargs)[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_token = out == eos_token

                if is_eos_token.any(dim=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        return out

    def forward(self, x, **kwargs):
        x_inp, x_labels = x[:, :-1], x[:, 1:]
        logits = self.net(x_inp, **kwargs)
        gd.debuginfo(prj="mt", info=f'x_inp={infoTensor(x_inp)}')
        gd.debuginfo(prj="mt", info=f'x_labels={infoTensor(x_labels)}')
        gd.debuginfo(prj="mt", info=f'logits={infoTensor(logits)}')
        return F.cross_entropy(rearrange(logits, "b c n -> b n c"), x_labels)
