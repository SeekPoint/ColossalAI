import datasets
import torch
import transformers
from args import parse_demo_args
from data import NetflixDataset, netflix_collator
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM, get_linear_schedule_with_warmup
from transformers.utils.versions import require_version
import os
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from pydebug import gd, infoTensor

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")
require_version("transformers>=4.20.0", "To fix: pip install -r requirements.txt")

output_transform_fn = lambda x: x
criterion = lambda x: x.loss

def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, dataloader, booster, coordinator):
    torch.cuda.synchronize()

    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    total_step = len(dataloader)
    gd.debuginfo(prj="mt", info=f'use_pipeline={use_pipeline}')
    gd.debuginfo(prj="mt", info=f'is_pp_last_stage={is_pp_last_stage}')
    gd.debuginfo(prj="mt", info=f'total_step={total_step}')
    model.train()
    gd.debuginfo(prj="mt", info=f'------------model.train done-----------------')
    optimizer.zero_grad()
    gd.debuginfo(prj="mt", info=f'------------optimizer.zero_grad done-----------------')
    dataloader = iter(dataloader)
    gd.debuginfo(prj="mt", info=f'dataloader={dataloader}')
    with tqdm(range(total_step),
              desc=f"Epoch [{epoch + 1}]",
              disable=not (coordinator.is_master() or is_pp_last_stage)) \
            as pbar:
        # Forward pass
        for step in pbar:
            if step > 10:
                break
            logff = f'Training_epoch{epoch:02}_step{step:02}'
            gd.emb_start(info=logff)   # 注意嵌套的重名问题
            if use_pipeline:
                gd.debuginfo(prj="mt", info=f'')
                outputs = booster.execute_pipeline(
                    dataloader, model, _criterion, optimizer, return_loss=True, return_outputs=True
                )
                gd.debuginfo(prj="mt", info=f'outputs={outputs}')
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    gd.debuginfo(prj="mt", info=f'loss={loss}')
                    pbar.set_postfix({"loss": loss.item()})
            else:
                gd.debuginfo(prj="mt", info=f'')
                data = next(dataloader)
                for k, v in data.items():
                    gd.debuginfo(prj="mt", info=f'1-data[{k}]={v}')
                data = move_to_cuda(data, device='cuda')
                for k, v in data.items():
                    gd.debuginfo(prj="mt", info=f'2-data[{k}]={v}')

                logf = f'model_forward_criterion_epoch{epoch:02}_step{step:02}'
                gd.emb_start(info=logf)
                outputs = model(**data)
                gd.debuginfo(prj="mt", info=f'outputs={outputs}')
                gd.debuginfo(prj="mt", info=f'++++++++++++++++++++++++++++++++++++++++++')
                loss = _criterion(outputs, None)
                gd.debuginfo(prj="mt", info=f'loss={loss}')
                gd.emb_end(info=logf)

                # Backward
                logf = f'boost_backward_epoch{epoch:02}_step{step:02}'
                gd.emb_start(info=logf)
                booster.backward(loss, optimizer)
                gd.emb_end(info=logf)
                pbar.set_postfix({"loss": loss.item()})

            gd.debuginfo(prj="mt", info=f'------------opt 1-----------------')
            logf = f'optimizer.step_epoch{epoch:02}_step{step:02}'
            gd.emb_start(info=logf)
            optimizer.step()
            gd.emb_end(info=logf)

            gd.debuginfo(prj="mt", info=f'------------opt 2-----------------')
            optimizer.zero_grad()

            gd.debuginfo(prj="mt", info=f'------------opt 3-----------------')
            lr_scheduler.step()

            gd.emb_end(info=logff)


def main():
    args = parse_demo_args()
    gd.debuginfo(prj="mt", info=f'args={args}')

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    gd.debuginfo(prj="mt", info=f'coordinator={coordinator}')
    gd.debuginfo(prj="mt", info=f'world_size={world_size}')

    # Manage loggers
    disable_existing_loggers()
    # logger = get_dist_logger()
    if coordinator.is_master():
        gd.debuginfo(prj="mt", info=f'')
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        gd.debuginfo(prj="mt", info=f'')
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logf = f'Initialize_OPT'
    gd.emb_start(info=logf)

    # Build OPT model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = OPTForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    gd.debuginfo(prj="mt", info=f'config={config}')
    gd.debuginfo(prj="mt", info=f'model={model}')

    gd.debuginfo(prj="mt", info=f"Finish loading model from {args.model_name_or_path}")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    gd.debuginfo(prj="mt", info=f'======================OPT 1=======================')

    # Set plugin
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        gd.debuginfo(prj="mt", info=f'')
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        gd.debuginfo(prj="mt", info=f'')
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        gd.debuginfo(prj="mt", info=f'')
        plugin = GeminiPlugin(offload_optim_frac=1.0, pin_memory=True, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        gd.debuginfo(prj="mt", info=f'')
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    elif args.plugin == "hybrid_parallel":
        gd.debuginfo(prj="mt", info=f'')
        # modify the param accordingly for finetuning test cases
        plugin = HybridParallelPlugin(
            tp_size=2,
            pp_size=1,
            num_microbatches=2,
            enable_all_optimization=True,
            zero_stage=0,
            precision="fp16",
            initial_scale=1,
        )

    gd.debuginfo(prj="mt", info=f"Set plugin as {args.plugin}")

    # Prepare tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    gd.debuginfo(prj="mt", info=f"tokenizer={tokenizer}")

    dataset = NetflixDataset(tokenizer)
    gd.debuginfo(prj="mt", info=f"dataset={dataset}")

    dataloader = plugin.prepare_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=netflix_collator
    )
    gd.debuginfo(prj="mt", info=f"dataloader={dataloader}")


    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=(args.learning_rate * world_size), weight_decay=args.weight_decay)
    gd.debuginfo(prj="mt", info=f"optimizer={optimizer}")

    # Set lr scheduler
    total_steps = len(dataloader) * args.num_epoch
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    gd.debuginfo(prj="mt", info=f"total_steps={total_steps}")
    gd.debuginfo(prj="mt", info=f"num_warmup_steps={num_warmup_steps}")

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(dataloader) * args.num_epoch
    )
    gd.debuginfo(prj="mt", info=f"lr_scheduler={lr_scheduler}")

    # Define criterion
    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    gd.debuginfo(prj="mt", info=f"booster={booster}")

    model, optimizer, _criterion, dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, dataloader=dataloader, criterion=_criterion, lr_scheduler=lr_scheduler
    )

    gd.debuginfo(prj="mt", info=f"model={model}")
    gd.debuginfo(prj="mt", info=f"optimizer={optimizer}")
    gd.debuginfo(prj="mt", info=f"_criterion={_criterion}")
    gd.debuginfo(prj="mt", info=f"dataloader={dataloader}")
    gd.debuginfo(prj="mt", info=f"lr_scheduler={lr_scheduler}")

    gd.emb_end(info=logf)


    # Start finetuning
    gd.debuginfo(prj="mt", info=f"Start finetuning")
    for epoch in range(args.num_epoch):
        train_epoch(epoch, model, optimizer, _criterion, lr_scheduler, dataloader, booster, coordinator)

    # Finish training and evaluate
    gd.debuginfo(prj="mt", info=f"Finish finetuning")

    logf = f'OPT_Save_model'
    gd.emb_start(info=logf)
    booster.save_model(model, args.output_path, shard=True)
    gd.emb_end(info=logf)

    gd.debuginfo(prj="mt", info=f"Saving model checkpoint to {args.output_path}")


if __name__ == "__main__":
    gd.debuginfo(prj='mt', info=f'=================') # 不被计入
    gd.setIgnore(prj='mt', ignore=20)

    gd.prjenable('ALL')  #打开项目flag

    logpath = f'/logColossalAI/_log_tmps_OPT_/'

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    gd.emb_mode(path=logpath, embedded_mode=True)

    main()

    gd.emb_mode(embedded_mode=False)
