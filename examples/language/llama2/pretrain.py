import argparse
import os
import resource
from contextlib import nullcontext
from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from attn import SUPPORT_XFORMERS, replace_xformers
from data_utils import load_json, prepare_dataloader, save_json
from datasets import load_dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from pydebug import gd, infoTensor

#修改这里减少模型大小，从而可以加大bs，最终减少时间

# ValueError: hidden_size must be divisible by num_heads (got `hidden_size`: 1024 and `num_heads`: 10).
MODEL_CONFIGS = {
    "7b": LlamaConfig(max_position_embeddings=128,
                      hidden_size = 64,
                      intermediate_size = 128,
                      num_hidden_layers = 5,
                      num_attention_heads = 2),
    "13b": LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        max_position_embeddings=4096,
    ),
    "70b": LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        max_position_embeddings=4096,
        num_key_value_heads=8,
    ),
}


def get_model_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def tokenize_batch_for_pretrain(batch, tokenizer: Optional[LlamaTokenizer] = None, max_length: int = 2048):
    texts = [sample["text"] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def save(
    booster: Booster,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    save_dir: str,
):
    save_dir = os.path.join(save_dir, f"epoch{epoch}-step{step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
    booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    running_states = {
        "epoch": epoch,
        "step": step,
        "sample_start_index": step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))


def load(
    booster: Booster, model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler, load_dir: str
) -> Tuple[int, int, int]:
    booster.load_model(model, os.path.join(load_dir, "model"))
    booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    return running_states["epoch"], running_states["step"], running_states["sample_start_index"]


def _criterion(outputs, inputs):
    return outputs.loss


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="7b", help="Model configuration")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "hybrid_parallel"],
        default="gemini",
        help="Choose which plugin to use",
    )

    #  465257 大概迭代数  3634-bs128 增大可bs缩短试验时间
    parser.add_argument(
        "-d", "--dataset", type=str, default="/share/hf_model/RedPajama-Data-1T-Sample", help="Data set path"
    )
    parser.add_argument("-e", "--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Local batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-w", "--weigth_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("-s", "--warmup_steps", type=int, default=2, help="Warmup steps")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("-x", "--mixed_precision", default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("-i", "--save_interval", type=int, default=2, help="Save interval")
    parser.add_argument("-o", "--save_dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("-f", "--load", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("-t", "--tensorboard_dir", type=str, default="tb_logs", help="Tensorboard directory")
    parser.add_argument("-a", "--flash_attention", action="store_true", help="Use Flash Attention")
    args = parser.parse_args()
    gd.debuginfo(prj="mt", info=f'args={args}')

    logf = f'Initialize_Distributed_Training'
    gd.emb_start(info=logf)

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({})
    gd.debuginfo(prj="mt", info=f'=====================llama 1==================================')
    coordinator = DistCoordinator()
    gd.debuginfo(prj="mt", info=f'coordinator={coordinator}')

    gd.emb_end(info=logf)

    logf = f'Initialize_Booster'
    gd.emb_start(info=logf)

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "gemini":
        plugin = GeminiPlugin(precision=args.mixed_precision, initial_scale=2**16, max_norm=args.grad_clip)
        gd.debuginfo(prj="mt", info=f'plugin={plugin}')
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision, placement_policy="auto", initial_scale=2**16, max_norm=args.grad_clip
        )
        gd.debuginfo(prj="mt", info=f'plugin={plugin}')
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, max_norm=args.grad_clip
        )
        gd.debuginfo(prj="mt", info=f'plugin={plugin}')
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, cpu_offload=True, max_norm=args.grad_clip
        )
        gd.debuginfo(prj="mt", info=f'plugin={plugin}')
    elif args.plugin == "hybrid_parallel":
        # modify the param accordingly, default configuration is for llama2-7b
        plugin = HybridParallelPlugin(
            tp_size=4,
            pp_size=2,
            num_microbatches=None,
            microbatch_size=1,
            enable_jit_fused=False,
            zero_stage=0,
            precision="fp32",
            initial_scale=1,
        )
        gd.debuginfo(prj="mt", info=f'plugin={plugin}')
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    gd.debuginfo(prj="mt", info=f'booster={booster}')

    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()

    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_stage)

    gd.debuginfo(prj="mt", info=f'use_pipeline={use_pipeline}')
    gd.debuginfo(prj="mt", info=f'is_pp_last_stage={is_pp_last_stage}')

    gd.emb_end(info=logf)

    # ==============================
    # Initialize Tensorboard
    # ==============================
    if print_flag:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)

    logf = f'Initialize-Tokenizer-Dataset-Dataloader'
    gd.emb_start(info=logf)

    # ==============================
    # Initialize Tokenizer, Dataset and Dataloader
    # ==============================
    tokenizer = LlamaTokenizer.from_pretrained("/share/hf_model/llama-tokenizer")
    # follows fast chat: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L257
    tokenizer.pad_token = tokenizer.unk_token

    gd.debuginfo(prj="mt", info=f'tokenizer={tokenizer}')

    dataset = load_dataset(args.dataset)
    gd.debuginfo(prj="mt", info=f'dataset={dataset}')

    gd.debuginfo(prj="mt", info=f'=====================llama 2==================================')

    train_ds = dataset["train"]
    gd.debuginfo(prj="mt", info=f'train_ds={train_ds}')

    dataloader = prepare_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(tokenize_batch_for_pretrain, tokenizer=tokenizer, max_length=args.max_length),
    )
    gd.debuginfo(prj="mt", info=f'dataloader={dataloader}')

    gd.debuginfo(prj="mt", info=f'=====================llama 3==================================')

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    config = MODEL_CONFIGS[args.config]
    gd.debuginfo(prj="mt", info=f'config={config}')

    # use lazy init when using GeminiPlugin
    init_ctx = (
        LazyInitContext(default_device=get_current_device()) if isinstance(plugin, GeminiPlugin) else nullcontext()
    )

    gd.debuginfo(prj="mt", info=f'init_ctx={init_ctx}')

    logf = f'llama2_model'
    gd.emb_start(info=logf)

    with init_ctx:
        model = LlamaForCausalLM(config)
        gd.debuginfo(prj="mt", info=f'model={model}')

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        gd.debuginfo(prj="mt", info=f'')
    if args.flash_attention:
        assert SUPPORT_XFORMERS, "Use flash attention while xfomers is not installed"
        replace_xformers(model)
        gd.debuginfo(prj="mt", info=f'')

    model_numel = get_model_numel(model)
    gd.debuginfo(prj="mt", info=f'model_numel={model_numel}')

    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")

    gd.debuginfo(prj="mt", info=f'=====================llama 4==================================')

    optimizer = HybridAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weigth_decay)
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=args.num_epochs * len(dataloader), warmup_steps=args.warmup_steps, eta_min=0.1 * args.lr
    )
    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)

    gd.debuginfo(prj="mt", info=f'=====================llama 5==================================')

    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
    )
    torch.set_default_dtype(torch.float)

    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )

    # load checkpoint if specified
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    if args.load is not None:
        coordinator.print_on_master("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx = load(booster, model, optimizer, lr_scheduler, args.load)
        coordinator.print_on_master(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")

    num_steps_per_epoch = len(dataloader)

    # if resume training, set the sampler start index to the correct value
    dataloader.sampler.set_start_index(sampler_start_idx)

    gd.emb_end(info=logf)

    for epoch in range(start_epoch, args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        step_nums = num_steps_per_epoch - start_step
        dataloader_iter = iter(dataloader)

        gd.debuginfo(prj="mt", info=f'step_nums={step_nums}')
        gd.debuginfo(prj="mt", info=f'dataloader_iter={dataloader_iter}')

        with tqdm(
            range(step_nums),
            desc=f"Epoch {epoch}",
            disable=not print_flag,
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                if step > 5:
                    break
                logf = f'epoch={epoch}+step={step}'
                gd.emb_start(info=logf)
                if use_pipeline:
                    outputs = booster.execute_pipeline(
                        dataloader_iter, model, _criterion, optimizer, return_loss=True, return_outputs=True
                    )
                    gd.debuginfo(prj="mt", info=f'outputs={outputs}')

                    loss = outputs["loss"]
                    gd.debuginfo(prj="mt", info=f'loss={loss}')
                else:
                    batch = next(dataloader_iter)
                    gd.debuginfo(prj="mt", info=f'batch["input_ids"]={batch["input_ids"]}')
                    gd.debuginfo(prj="mt", info=f'batch["labels"]={batch["labels"]}')

                    outputs = model(**batch)
                    gd.debuginfo(prj="mt", info=f'outputs={outputs}')

                    loss = outputs[0]
                    gd.debuginfo(prj="mt", info=f'loss={loss}')

                    booster.backward(loss, optimizer)
                gd.debuginfo(prj="mt", info=f'=====================llama 6==================================')
                optimizer.step()
                gd.debuginfo(prj="mt", info=f'=====================llama 7==================================')
                lr_scheduler.step()
                gd.debuginfo(prj="mt", info=f'=====================llama 8==================================')
                optimizer.zero_grad()
                gd.debuginfo(prj="mt", info=f'=====================llama 9==================================')

                if not use_pipeline:
                    all_reduce_mean(loss)
                    gd.debuginfo(prj="mt", info=f'')

                if print_flag:
                    pbar.set_postfix({"loss": loss.item()})
                    writer.add_scalar("loss", loss.item(), epoch * num_steps_per_epoch + step)

                if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                    gd.debuginfo(prj="mt", info=f'')
                    coordinator.print_on_master(f"Saving checkpoint")
                    save(
                        booster,
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        args.batch_size,
                        coordinator,
                        args.save_dir,
                    )
                    coordinator.print_on_master(f"Saved checkpoint at epoch {epoch} step {step + 1}")
                gd.emb_end(info=logf)

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0
        gd.debuginfo(prj="mt", info=f'set start_step to 0')

    coordinator.print_on_master(f"Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    gd.debuginfo(prj='mt', info=f'=================') # 不被计入

    gd.prjenable('ALL')  #打开项目flag

    logpath = f'/workspace/yk_repo/ColossalAI/_log_tmps_llama2_/'

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    gd.emb_mode(path=logpath, embedded_mode=True)

    main()

    gd.emb_mode(embedded_mode=False)

