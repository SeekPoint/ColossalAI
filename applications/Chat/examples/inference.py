import argparse

import torch
from coati.models.bloom import BLOOMActor
from coati.models.generation import generate
from coati.models.gpt import GPTActor
from coati.models.llama import LlamaActor
from coati.models.opt import OPTActor
from transformers import AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, LlamaTokenizer
from pydebug import gd, infoTensor

def eval(args):
    # configure model
    if args.model == "gpt2":
        gd.debuginfo(prj="mt", info=f'')
        actor = GPTActor(pretrained=args.pretrain)
    elif args.model == "bloom":
        gd.debuginfo(prj="mt", info=f'')
        actor = BLOOMActor(pretrained=args.pretrain)
    elif args.model == "opt":
        gd.debuginfo(prj="mt", info=f'')
        actor = OPTActor(pretrained=args.pretrain)
    elif args.model == "llama":
        gd.debuginfo(prj="mt", info=f'')
        actor = LlamaActor(pretrained=args.pretrain)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    actor.to(torch.cuda.current_device())
    if args.model_path is not None:
        gd.debuginfo(prj="mt", info=f'')
        state_dict = torch.load(args.model_path)
        actor.load_state_dict(state_dict)

    # configure tokenizer
    if args.model == "gpt2":
        gd.debuginfo(prj="mt", info=f'')
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "bloom":
        gd.debuginfo(prj="mt", info=f'')
        tokenizer = BloomTokenizerFast.from_pretrained("/share/hf_model/bloom-560m")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "opt":
        gd.debuginfo(prj="mt", info=f'')
        tokenizer = AutoTokenizer.from_pretrained("/share/hf_model/opt-350m")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "llama":
        gd.debuginfo(prj="mt", info=f'')
        tokenizer = LlamaTokenizer.from_pretrained("/share/hf_model/llama-tokenizer")
        tokenizer.eos_token = "<\s>"
        tokenizer.pad_token = tokenizer.unk_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    actor.eval()
    tokenizer.padding_side = "left"
    input_ids = tokenizer.encode(args.input, return_tensors="pt").to(torch.cuda.current_device())
    outputs = generate(
        actor,
        input_ids,
        tokenizer=tokenizer,
        max_length=args.max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    print(f"[Output]: {''.join(output)}")


if __name__ == "__main__":
    gd.debuginfo(prj="mt", info=f'')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "bloom", "opt", "llama"])
    # We suggest to use the pretrained model from HuggingFace, use pretrain to configure model
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--input", type=str, default="Question: How are you ? Answer:")
    parser.add_argument("--max_length", type=int, default=100)
    args = parser.parse_args()
    eval(args)
