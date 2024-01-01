import argparse
import warnings
import sys
sys.path.append('./')
import os
import torch
import torch.distributed as dist
from coati.dataset import HhRlhfDataset, RmStaticDataset
from coati.models import LogExpLoss, LogSigLoss
from coati.models.bloom import BLOOMRM
from coati.models.gpt import GPTRM
from coati.models.llama import LlamaRM
from coati.models.opt import OPTRM
from coati.trainer import RewardModelTrainer
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from datasets import load_dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomTokenizerFast, LlamaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam
from pydebug import gd, infoTensor

def train(args):
    # configure strategy
    if args.strategy == "ddp":
        strategy = DDPStrategy()
        gd.debuginfo(prj="mt", info=f'')
    elif args.strategy == "colossalai_gemini":
        gd.debuginfo(prj="mt", info=f'')
        strategy = GeminiStrategy(placement_policy="auto")
    elif args.strategy == "colossalai_zero2":
        strategy = LowLevelZeroStrategy(stage=2, placement_policy="cuda")
        gd.debuginfo(prj="mt", info=f'')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    if args.lora_rank > 0:
        gd.debuginfo(prj="mt", info=f"Lora is not supported yet.")
        args.lora_rank = 0

    with strategy.model_init_context():
        if args.model == "bloom":
            model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
            gd.debuginfo(prj="mt", info=f'')
        elif args.model == "opt":
            model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
            gd.debuginfo(prj="mt", info=f'')
        elif args.model == "gpt2":
            model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
            gd.debuginfo(prj="mt", info=f'')
        elif args.model == "llama":
            model = LlamaRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
            gd.debuginfo(prj="mt", info=f'')
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

        model.to(torch.bfloat16).to(torch.cuda.current_device())

        if args.model_path is not None:
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)

    # configure tokenizer
    if args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2" if args.tokenizer is None else args.tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        gd.debuginfo(prj="mt", info=f'')
    elif args.model == "bloom":
        tokenizer = BloomTokenizerFast.from_pretrained(
            "/share/hf_model/bloom-560m" if args.tokenizer is None else args.tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
        gd.debuginfo(prj="mt", info=f'')
    elif args.model == "opt":
        tokenizer = AutoTokenizer.from_pretrained("/share/hf_model/opt-350m" if args.tokenizer is None else args.tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        gd.debuginfo(prj="mt", info=f'')
    elif args.model == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(
            "/share/hf_model/llama-tokenizer" if args.tokenizer is None else args.tokenizer
        )
        tokenizer.eos_token = "<\s>"
        tokenizer.pad_token = tokenizer.unk_token
        gd.debuginfo(prj="mt", info=f'')
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    # configure optimizer
    if args.strategy.startswith("colossalai"):
        optim = HybridAdam(model.parameters(), lr=args.lr)
        gd.debuginfo(prj="mt", info=f'')
    else:
        optim = Adam(model.parameters(), lr=args.lr)
        gd.debuginfo(prj="mt", info=f'')

    # configure loss function
    if args.loss_fn == "log_sig":
        loss_fn = LogSigLoss()
        gd.debuginfo(prj="mt", info=f'')
    elif args.loss_fn == "log_exp":
        loss_fn = LogExpLoss()
        gd.debuginfo(prj="mt", info=f'')
    else:
        raise ValueError(f'Unsupported loss function "{args.loss_fn}"')

    print(f"args.dataset={args.dataset}")
    #"Anthropic/hh-rlhf", "Dahoas/rm-static"

    if args.dataset == "Anthropic/hh-rlhf":
        data_path = '/share/hf_model/hh-rlhf'
    elif args.dataset == "Dahoas/rm-static":
        data_path = '/share/hf_model/rm-static'
    else:
        data_path = args.dataset
    gd.debuginfo(prj="mt", info=f'data_path={data_path}')

    # prepare for data and dataset
    if args.subset is not None:
        data = load_dataset(data_path, data_dir=args.subset)
        gd.debuginfo(prj="mt", info=f'')
    else:
        data = load_dataset(data_path)
        gd.debuginfo(prj="mt", info=f'')

    train_data = data["train"].select(range(min(args.max_datasets_size, len(data["train"]))))
    eval_data = data["test"].select(range(min(args.max_datasets_size, len(data["test"]))))

    if args.dataset == "Dahoas/rm-static":
        train_dataset = RmStaticDataset(train_data, tokenizer, args.max_len)
        eval_dataset = RmStaticDataset(eval_data, tokenizer, args.max_len)
        gd.debuginfo(prj="mt", info=f'')
    elif args.dataset == "Anthropic/hh-rlhf":
        train_dataset = HhRlhfDataset(train_data, tokenizer, args.max_len)
        eval_dataset = HhRlhfDataset(eval_data, tokenizer, args.max_len)
        gd.debuginfo(prj="mt", info=f'')
    else:
        raise ValueError(f'Unsupported dataset "{args.dataset}"')

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=42,
            drop_last=True,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            shuffle=True,
            seed=42,
            drop_last=True,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
        )
    else:
        train_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        eval_dataset, shuffle=(eval_sampler is None), sampler=eval_sampler, batch_size=args.batch_size, pin_memory=True
    )

    lr_scheduler = CosineAnnealingLR(optim, train_dataloader.__len__() // 100)
    strategy_dict = strategy.prepare(dict(model=model, optimizer=optim, lr_scheduler=lr_scheduler))
    model = strategy_dict["model"]
    optim = strategy_dict["optimizer"]
    lr_scheduler = strategy_dict["lr_scheduler"]
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        max_epochs=args.max_epochs,
    )

    trainer.fit(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
    )

    if args.lora_rank > 0 and args.merge_lora_weights:
        from coati.models.lora import LORA_MANAGER

        # NOTE: set model to eval to merge LoRA weights
        LORA_MANAGER.merge_weights = True
        model.eval()
    # save model checkpoint after fitting on only rank0
    state_dict = model.state_dict()
    torch.save(state_dict, args.save_path)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(
            trainer.optimizer, "rm_optim_checkpoint_%d.pt" % (torch.cuda.current_device()), only_rank0=False
        )


if __name__ == "__main__":
    gd.debuginfo(prj='mt', info=f'=================') # 不被计入
    gd.setIgnore(prj='mt', ignore=20)

    gd.prjenable('ALL')  #打开项目flag

    logpath = f'/workspace/yk_repo/ColossalAI/_log_tmps_chat_RM_/'
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    gd.emb_mode(path=logpath, embedded_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", choices=["ddp", "colossalai_gemini", "colossalai_zero2"], default="colossalai_zero2"
    )
    parser.add_argument("--model", choices=["gpt2", "bloom", "opt", "llama"], default="bloom")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--need_optim_ckpt", type=bool, default=False)
    parser.add_argument(
        "--dataset", type=str, choices=["Anthropic/hh-rlhf", "Dahoas/rm-static"], default="Dahoas/rm-static"
    )
    parser.add_argument("--subset", type=lambda x: None if x == "None" else x, default=None)
    parser.add_argument("--max_datasets_size", type=int, default=3000)
    parser.add_argument("--save_path", type=str, default="rm_ckpt")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument("--merge_lora_weights", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=9e-6)
    parser.add_argument("--loss_fn", type=str, default="log_sig", choices=["log_sig", "log_exp"])
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    args = parser.parse_args()
    gd.debuginfo(prj="mt", info=f'args={args}')

    train(args)

    gd.emb_mode(embedded_mode=False)
