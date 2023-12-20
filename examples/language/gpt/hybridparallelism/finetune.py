import argparse
from typing import Callable, List, Union
import os
import evaluate
import torch
import torch.distributed as dist
import torch.nn as nn
from data import GLUEDataBuilder
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from pydebug import gd, infoTensor

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

output_transform_fn = lambda x: x
criterion = lambda x: x.loss


def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    criterion,
    test_dataloader: Union[DataLoader, List[DataLoader]],
    num_labels: int,
    task_name: str,
    eval_splits: List[str],
    booster: Booster,
    coordinator: DistCoordinator,
):
    metric = evaluate.load("/share/hf_eval/glue", task_name, process_id=coordinator.rank, num_process=coordinator.world_size)
    gd.debuginfo(prj="mt", info=f'metric={metric}')

    model.eval()

    gd.debuginfo(prj="mt", info=f'=======================================')

    def evaluate_subset(dataloader: DataLoader):
        use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
        is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()

        gd.debuginfo(prj="mt", info=f'use_pipeline={use_pipeline}')
        gd.debuginfo(prj="mt", info=f'is_pp_last_stage={is_pp_last_stage}')

        accum_loss = torch.zeros(1, device=get_current_device())
        gd.debuginfo(prj="mt", info=f'accum_loss={accum_loss}')

        for i, batch in enumerate(dataloader):
            logf = f'evaluate_model_OPT_batch{i:03}'
            gd.emb_start(info=logf)
            batch = move_to_cuda(batch)
            gd.debuginfo(prj="mt", info=f'The {i}th batch={batch}')
            labels = batch["labels"]
            if use_pipeline:
                pg_mesh = booster.plugin.pg_mesh
                pp_group = booster.plugin.pp_group
                current_pp_group_ranks = pg_mesh.get_ranks_in_group(pp_group)
                current_rank = dist.get_rank()
                batch = iter([batch])
                outputs = booster.execute_pipeline(batch, model, criterion, return_loss=True, return_outputs=True)
                gd.debuginfo(prj="mt", info=f'pg_mesh={pg_mesh}')
                gd.debuginfo(prj="mt", info=f'pp_group={pp_group}')
                gd.debuginfo(prj="mt", info=f'current_pp_group_ranks={current_pp_group_ranks}')
                gd.debuginfo(prj="mt", info=f'current_rank={current_rank}')
                gd.debuginfo(prj="mt", info=f'batch={batch}')
                gd.debuginfo(prj="mt", info=f'outputs={outputs}')

                if is_pp_last_stage:
                    logits = outputs["outputs"]["logits"]
                    gd.debuginfo(prj="mt", info=f'logits={infoTensor(logits)}')

                    val_loss = outputs["loss"]
                    gd.debuginfo(prj="mt", info=f'val_loss={val_loss}')

                    accum_loss.add_(val_loss)

                    if num_labels > 1:
                        preds = torch.argmax(logits, axis=1)
                        gd.debuginfo(prj="mt", info=f'preds={preds}')
                    elif num_labels == 1:
                        preds = logits.squeeze()
                        gd.debuginfo(prj="mt", info=f'preds={preds}')

                    dist.broadcast_object_list([preds, val_loss], src=current_pp_group_ranks[-1], group=pp_group)

                    metric.add_batch(predictions=preds, references=labels)
                elif current_rank in current_pp_group_ranks:
                    gd.debuginfo(prj="mt", info=f'---------------------------')
                    object_list = [None, None]
                    dist.broadcast_object_list(object_list, src=current_pp_group_ranks[-1], group=pp_group)
                    gd.debuginfo(prj="mt", info=f'--------------------------')
                    metric.add_batch(predictions=object_list[0].to(get_current_device()), references=labels)
                    accum_loss.add_(object_list[1].to(get_current_device()))

            else:
                batch = move_to_cuda(batch)
                outputs = model(**batch)
                val_loss, logits = outputs[:2]
                accum_loss.add_(val_loss)

                gd.debuginfo(prj="mt", info=f'val_loss={val_loss}')
                gd.debuginfo(prj="mt", info=f'logits={infoTensor(logits)}')
                gd.debuginfo(prj="mt", info=f'outputs={outputs}')


                if num_labels > 1:
                    preds = torch.argmax(logits, axis=1)
                    gd.debuginfo(prj="mt", info=f'preds={preds}')
                elif num_labels == 1:
                    preds = logits.squeeze()
                    gd.debuginfo(prj="mt", info=f'preds={preds}')

                metric.add_batch(predictions=preds, references=labels)

            gd.emb_end(info=logf)

        results = metric.compute()
        gd.debuginfo(prj="mt", info=f'results={results}')
        dist.all_reduce(accum_loss.div_(len(dataloader)))
        gd.debuginfo(prj="mt", info=f'+++++++++++++++++++++++++++++++++++++++++++')
        if coordinator.is_master() and results is not None:
            gd.debuginfo(prj="mt", info=f'')
            results["loss"] = accum_loss.item() / coordinator.world_size

        return results

    if isinstance(test_dataloader, DataLoader):
        gd.debuginfo(prj="mt", info=f'')
        return evaluate_subset(test_dataloader)
    else:
        assert len(test_dataloader) == len(eval_splits)
        final_results = {}
        for split, sub_loader in zip(eval_splits, test_dataloader):
            gd.debuginfo(prj="mt", info=f'split={split}, sub_loader={sub_loader}')
            results = evaluate_subset(sub_loader)
            final_results.update({f"{k}_{split}": v for k, v in results.items()})
        return final_results


def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    _criterion: Callable,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    total_step = len(train_dataloader)

    gd.debuginfo(prj="mt", info=f'use_pipeline={use_pipeline}')
    gd.debuginfo(prj="mt", info=f'is_pp_last_stage={is_pp_last_stage}')
    gd.debuginfo(prj="mt", info=f'total_step={total_step}')


    model.train()
    gd.debuginfo(prj="mt", info=f'=================GPT HP 2==============================')

    optimizer.zero_grad()
    gd.debuginfo(prj="mt", info=f'=================GPT HP 3==============================')

    train_dataloader_iter = iter(train_dataloader)
    gd.debuginfo(prj="mt", info=f'train_dataloader_iter={train_dataloader_iter}')

    with tqdm(
        range(total_step),
        desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]",
        disable=not (coordinator.is_master() or is_pp_last_stage),
    ) as pbar:
        # Forward pass
        for step in pbar:
            if step > 10:
                break
            logff = f'Train_GPT_HBP_epoch{epoch:02}_step{step:02}'
            gd.emb_start(info=logff)   # 注意嵌套的重名问题

            if use_pipeline:
                outputs = booster.execute_pipeline(
                    train_dataloader_iter, model, _criterion, optimizer, return_loss=True, return_outputs=True
                )
                gd.debuginfo(prj="mt", info=f'outputs={outputs}')
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})
            else:
                data = next(train_dataloader_iter)
                for k, v in data.items():
                    gd.debuginfo(prj="mt", info=f'1-data[{k}]={v}')

                data = move_to_cuda(data)
                for k, v in data.items():
                    gd.debuginfo(prj="mt", info=f'2-data[{k}]={v}')

                logf = f'model_forward_criterion_epoch{epoch:02}_step{step:02}'
                gd.emb_start(info=logf)

                outputs = model(**data)
                gd.debuginfo(prj="mt", info=f'outputs={outputs}')

                loss = _criterion(outputs, None)
                gd.debuginfo(prj="mt", info=f'loss={loss}')

                gd.emb_end(info=logf)

                # Backward
                logf = f'boost_backward_epoch{epoch:02}_step{step:04}'
                gd.emb_start(info=logf)
                booster.backward(loss, optimizer)
                gd.emb_end(info=logf)

                gd.debuginfo(prj="mt", info=f'=================GPT HP 4==============================')
                pbar.set_postfix({"loss": loss.item()})

            gd.debuginfo(prj="mt", info=f'=================GPT HP 5==============================')

            logf = f'optimizer_step_epoch{epoch:02}'
            gd.emb_start(info=logf)
            optimizer.step()
            gd.emb_end(info=logf)

            gd.debuginfo(prj="mt", info=f'=================GPT HP 6==============================')
            optimizer.zero_grad()
            gd.debuginfo(prj="mt", info=f'=================GPT HP 7==============================')
            lr_scheduler.step()

            gd.emb_end(info=logff)


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default="mrpc", help="GLUE task to run")
    parser.add_argument(
        "-p",
        "--plugin",
        type=str,
        default="torch_ddp",
        choices=["torch_ddp", "torch_ddp_fp16", "gemini", "low_level_zero", "hybrid_parallel"],
        help="plugin to use",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        help="only gpt2 now",
    )
    parser.add_argument("--target_f1", type=float, default=None, help="target f1 score. Raise exception if not reached")
    parser.add_argument("--use_lazy_init", type=bool, default=False, help="for initiating lazy init context")
    args = parser.parse_args()
    gd.debuginfo(prj="mt", info=f'args={args}')

    if args.model_type == "gpt2":
        model_name = "/share/hf_model/gpt2"
    else:
        raise RuntimeError
    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(config={}, seed=42)
    gd.debuginfo(prj="mt", info=f'=================GPT HP 1==============================')

    coordinator = DistCoordinator()
    gd.debuginfo(prj="mt", info=f'coordinator={coordinator}')

    # local_batch_size = BATCH_SIZE // coordinator.world_size
    lr = LEARNING_RATE * coordinator.world_size
    gd.debuginfo(prj="mt", info=f'lr={lr}')


    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        gd.debuginfo(prj="mt", info=f'')
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        gd.debuginfo(prj="mt", info=f'')
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        gd.debuginfo(prj="mt", info=f'')
        plugin = GeminiPlugin(initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        gd.debuginfo(prj="mt", info=f'')
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    elif args.plugin == "hybrid_parallel":
        gd.debuginfo(prj="mt", info=f'')
        # modify the param accordingly for finetuning test cases
        plugin = HybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            num_microbatches=None,
            microbatch_size=1,
            enable_all_optimization=True,
            zero_stage=1,
            precision="fp16",
            initial_scale=1,
        )
    gd.debuginfo(prj="mt", info=f'plugin={plugin}')

    booster = Booster(plugin=plugin, **booster_kwargs)
    gd.debuginfo(prj="mt", info=f'booster={booster}')

    # ==============================
    # Prepare Dataloader
    # ==============================
    data_builder = GLUEDataBuilder(
        model_name, plugin, args.task, train_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE
    )
    gd.debuginfo(prj="mt", info=f'data_builder={data_builder}')

    train_dataloader = data_builder.train_dataloader()
    gd.debuginfo(prj="mt", info=f'train_dataloader={train_dataloader}')

    test_dataloader = data_builder.test_dataloader()
    gd.debuginfo(prj="mt", info=f'test_dataloader={test_dataloader}')

    # ====================================
    # Prepare model, optimizer
    # ====================================
    # gpt2 pretrained model

    cfg = AutoConfig.from_pretrained(model_name, num_labels=data_builder.num_labels)


    # https://stackoverflow.com/questions/68084302/assertionerror-cannot-handle-batch-sizes-1-if-no-padding-token-is-defined
    # "AssertionError: Cannot handle batch sizes > 1 if no padding token is > defined" and pad_token = eos_token
    cfg.pad_token_id = cfg.eos_token_id
    gd.debuginfo(prj="mt", info=f'cfg={cfg}')

    logf = f'GPT2ForSequenceClassification_model'
    gd.emb_start(info=logf)

    if model_name == "/share/hf_model/gpt2":
        model = GPT2ForSequenceClassification.from_pretrained(model_name, config=cfg).cuda()
    else:
        raise RuntimeError

    gd.debuginfo(prj="mt", info=f'model={model}')

    gd.emb_end(info=logf)

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    gd.debuginfo(prj="mt", info=f'optimizer={optimizer}')

    # lr scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(WARMUP_FRACTION * total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    gd.debuginfo(prj="mt", info=f'total_steps={total_steps}')
    gd.debuginfo(prj="mt", info=f'num_warmup_steps={num_warmup_steps}')
    gd.debuginfo(prj="mt", info=f'lr_scheduler={lr_scheduler}')

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, _criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=_criterion, lr_scheduler=lr_scheduler
    )

    logf = f'gpt_hybirdparallel_model'
    gd.emb_start(info=logf)

    gd.debuginfo(prj="mt", info=f'model={model}')
    gd.debuginfo(prj="mt", info=f'optimizer={optimizer}')
    gd.debuginfo(prj="mt", info=f'_criterion={_criterion}')
    gd.debuginfo(prj="mt", info=f'lr_scheduler={lr_scheduler}')

    gd.emb_end(info=logf)

    # ==============================
    # Train model
    # ==============================
    for epoch in range(NUM_EPOCHS):
        train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, train_dataloader, booster, coordinator)

    logf = f'evaluate_model_OPT_epoch'
    gd.emb_start(info=logf)
    results = evaluate_model(
        model,
        _criterion,
        test_dataloader,
        data_builder.num_labels,
        args.task,
        data_builder.eval_splits,
        booster,
        coordinator,
    )
    gd.debuginfo(prj="mt", info=f'results={results}')
    gd.emb_end(info=logf)

    if coordinator.is_master():
        # print(results)
        if args.target_f1 is not None and "f1" in results:
            assert results["f1"] >= args.target_f1, f'f1 score {results["f1"]} is lower than target {args.target_f1}'


if __name__ == "__main__":
    gd.debuginfo(prj='mt', info=f'=================')  # 不被计入

    gd.prjenable('ALL')  # 打开项目flag

    logpath = f'/workspace/yk_repo/ColossalAI/_log_tmps_GPT_hybirdParallel_/'

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    gd.emb_mode(path=logpath, embedded_mode=True)

    main()

    gd.emb_mode(embedded_mode=False)
