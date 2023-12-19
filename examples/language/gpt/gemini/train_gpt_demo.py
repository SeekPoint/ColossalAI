import argparse
import os
from contextlib import nullcontext
from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from commons.model_zoo import model_builder
from commons.utils import get_data, get_profile_context, get_tflops, get_time_stamp
from packaging import version

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.lazy import LazyInitContext
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from pydebug import gd, infoTensor

CAI_VERSION = colossalai.__version__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distplan",
        type=str,
        default="CAI_Gemini",
        help="The distributed plan [colossalai, zero1, zero2, torch_ddp, torch_zero].",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per DP group of training.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2_medium",
        help="model model scale",
    )
    parser.add_argument(
        "--train_step",
        type=int,
        default=10,
        help="training iterations for test",
    )

    args = parser.parse_args()
    return args


class GPTLMLoss(nn.Module):
    def __init__(self):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        gd.debuginfo(prj="mt", info=f'shift_logits={infoTensor(shift_logits)}')
        gd.debuginfo(prj="mt", info=f'shift_labels={infoTensor(shift_labels)}')
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=""):
    return f"{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB"


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f"{numel / GB_SIZE:.1f}B"
    elif numel >= MB_SIZE:
        return f"{numel / MB_SIZE:.1f}M"
    elif numel >= KB_SIZE:
        return f"{numel / KB_SIZE:.1f}K"
    else:
        return str(numel)


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split("\n")[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    gd.debuginfo(prj="mt", info=f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


def main():
    # version check
    # this example is supposed to work for versions greater than 0.2.0
    assert version.parse(CAI_VERSION) >= version.parse("0.2.0")

    set_cpu_maximum_parallelism()
    gd.debuginfo(prj="mt", info=f'=================GPT 1==============================')

    args = parse_args()
    gd.debuginfo(prj="mt", info=f'args={args}')

    # if args.distplan not in ["colossalai", "torch_ddp", "torch_zero", "zero1", "zero2"]:
    if args.distplan not in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]:
        raise TypeError(f"{args.distplan} is error")

    # batch size per DP degree
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257

    NUM_STEPS = args.train_step

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median"
    PROF_FLAG = False  # The flag of profiling, False by default

    disable_existing_loggers()

    gd.debuginfo(prj="mt", info=f'=================GPT 2==============================')
    colossalai.launch_from_torch(config={})

    # logger = get_dist_logger()
    gd.debuginfo(prj="mt", info=f"{args.model_type}, {args.distplan}, batch size {BATCH_SIZE}")

    # build criterion
    criterion = GPTLMLoss()
    torch.manual_seed(123)

    if args.distplan.startswith("CAI"):
        ctx = LazyInitContext(default_device=get_current_device()) if args.distplan == "CAI_Gemini" else nullcontext()
        gd.debuginfo(prj="mt", info=f'ctx={ctx}')

        # build GPT model
        with ctx:
            model = model_builder(args.model_type)(checkpoint=True)
            gd.debuginfo(prj="mt", info=f'model={model}')

        # assign running configurations
        if args.distplan == "CAI_ZeRO1":
            gd.debuginfo(prj="mt", info=f'')
            zero_stage = 1
        elif args.distplan == "CAI_ZeRO2":
            gd.debuginfo(prj="mt", info=f'')
            zero_stage = 2
        elif args.distplan == "CAI_Gemini":
            gd.debuginfo(prj="mt", info=f'')
            zero_stage = 3
        else:
            raise RuntimeError

        plugin = None
        if args.distplan.startswith("CAI_ZeRO"):
            plugin = LowLevelZeroPlugin(
                stage=zero_stage, reduce_bucket_size_in_m=12, overlap_communication=True, verbose=True
            )
            gd.debuginfo(prj="mt", info=f'plugin={plugin}')
        elif args.distplan == "CAI_Gemini":
            plugin = GeminiPlugin(search_range_m=128, hidden_dim=model.config.n_embd)
            gd.debuginfo(prj="mt", info=f'plugin={plugin}')
        else:
            raise RuntimeError

        # build a highly optimized gpu/cpu optimizer
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        gd.debuginfo(prj="mt", info=f'optimizer={optimizer}')

        gd.debuginfo(prj="mt", info=f'{get_mem_info(prefix="After init optim, ")}')
    elif args.distplan.startswith("Pytorch"):
        assert args.tp_degree == 1, "The degree of TP should be 1 for DDP examples."
        model = model_builder(args.model_type)(checkpoint=True).cuda()
        gd.debuginfo(prj="mt", info=f'model={model}')

        plugin = TorchDDPPlugin()
        gd.debuginfo(prj="mt", info=f'plugin={plugin}')

        if args.distplan.endswith("DDP"):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            gd.debuginfo(prj="mt", info=f'optimizer={optimizer}')
        elif args.distplan.endswith("ZeRO"):
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=1e-3)
            gd.debuginfo(prj="mt", info=f'optimizer={optimizer}')

    else:
        raise RuntimeError
    # wrap your model and optimizer
    booster = Booster(plugin=plugin)
    gd.debuginfo(prj="mt", info=f'booster={booster}')

    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    logf = f'gpt_gemini_model'
    gd.emb_start(info=logf)

    gd.debuginfo(prj="mt", info=f'model={model}')
    gd.debuginfo(prj="mt", info=f'optimizer={optimizer}')
    gd.debuginfo(prj="mt", info=f'criterion={criterion}')

    gd.emb_end(info=logf)


    # model is shared after TP
    numel = get_model_size(model)
    gd.debuginfo(prj="mt", info=f"the size of testing model size is {model_size_formatter(numel)}.")
    gd.debuginfo(prj="mt", info=f'{get_mem_info(prefix="After init model, ")}')

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)
    gd.debuginfo(prj="mt", info=f'get_tflops_func={get_tflops_func}')

    torch.cuda.synchronize()

    gd.debuginfo(prj="mt", info=f'=================GPT 1======================================')

    model.train()

    gd.debuginfo(prj="mt", info=f'=================GPT 2======================================')

    tflops_list = []

    def train_step():
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        gd.debuginfo(prj="mt", info=f'input_ids={input_ids}')
        gd.debuginfo(prj="mt", info=f'attn_mask={attn_mask}')

        optimizer.zero_grad()
        gd.debuginfo(prj="mt", info=f'=================GPT 3======================================')

        start = time()
        outputs = model(input_ids, attn_mask)
        gd.debuginfo(prj="mt", info=f'outputs={outputs}')

        loss = criterion(outputs, input_ids)
        torch.cuda.synchronize()
        gd.debuginfo(prj="mt", info=f'loss={loss}')

        fwd_end = time()
        fwd_time = fwd_end - start
        gd.debuginfo(prj="mt", info=f'{get_mem_info(prefix=f"[{n + 1}/{NUM_STEPS}] Forward ")}')

        booster.backward(loss, optimizer)

        torch.cuda.synchronize()
        gd.debuginfo(prj="mt", info=f'=================GPT 4======================================')

        bwd_end = time()
        bwd_time = bwd_end - fwd_end
        gd.debuginfo(prj="mt", info=f'{get_mem_info(prefix=f"[{n + 1}/{NUM_STEPS}] Backward ")}')

        optimizer.step()
        torch.cuda.synchronize()
        gd.debuginfo(prj="mt", info=f'=================GPT 5======================================')

        optim_time = time() - bwd_end
        step_time = time() - start
        gd.debuginfo(prj="mt", info=f'{get_mem_info(prefix=f"[{n + 1}/{NUM_STEPS}] Optimizer step ")}')

        step_tflops = get_tflops_func(step_time)

        gd.debuginfo(prj="mt", info=
            f"[{n + 1}/{NUM_STEPS}] "
            f"Loss:{loss.item():.3f}, "
            f"Step time: {step_time:.3f}s, "
            f"TFLOPS: {get_tflops_func(step_time):.3f}, "
            f"FWD time: {fwd_time:.3f}s, "
            f"BWD time: {bwd_time:.3f}s, "
            f"OPTIM time: {optim_time:.3f}s")

        if n >= WARMUP_STEPS:
            tflops_list.append(step_tflops)

    demo_profiler = get_profile_context(
        PROF_FLAG, WARMUP_STEPS, NUM_STEPS - WARMUP_STEPS, save_dir=f"profile/{get_time_stamp()}-demo"
    )

    with demo_profiler as prof:
        for n in range(NUM_STEPS):
            if n > 10:
                break
            logf = f'Train_OPT_step{n:4}'
            gd.emb_start(info=logf)
            train_step()
            gd.debuginfo(prj="mt", info=f'=================GPT 3======================================')
            prof.step()
            gd.emb_end(info=logf)

    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    gd.debuginfo(prj="mt", info=f"Median TFLOPS is {tflops_list[median_index]:.3f}")
    torch.cuda.synchronize()


if __name__ == "__main__":
    gd.debuginfo(prj='mt', info=f'=================') # 不被计入

    gd.prjenable('ALL')  #打开项目flag

    logpath = f'/workspace/yk_repo/ColossalAI/_log_tmps_GPT_gemimi_/'

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    gd.emb_mode(path=logpath, embedded_mode=True)

    main()
