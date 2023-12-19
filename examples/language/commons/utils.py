import torch
from pydebug import gd, infoTensor

# Randomly Generated Data
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    gd.debuginfo(prj="mt", info=f'input_ids={infoTensor(input_ids)}')

    attention_mask = torch.ones_like(input_ids)
    gd.debuginfo(prj="mt", info=f'attention_mask={infoTensor(attention_mask)}')

    return input_ids, attention_mask


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)
