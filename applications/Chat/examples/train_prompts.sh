set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv |
        tail -n +2 |
        nl -v 0 |
        tee /dev/tty |
        sort -g -k 2 |
        awk '{print $1}' |
        head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 2

# torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy colossalai_zero2

torchrun --standalone --nproc_per_node=3 /workspace/yk_repo/ColossalAI/applications/Chat/examples/train_prompts.py \
    --prompt_dataset /share/hf_model/instinwild_ch.json \
    --pretrain_dataset /share/hf_model/instinwild_ch.json \
    --strategy colossalai_zero2 \
    --pretrain Coati-bloom-560m-sft \
    --save_path Coati-bloom-560m-rl \
    --model 'bloom' \
    --rm_pretrain Coati-bloom-560m-sft \
    --rm_path Coati-bloom-560m-rw.pt \
    --max_datasets_size 500 \
    --num_episodes 1

# 前面两个阶段就OOM
#torchrun --standalone --nproc_per_node=4 train_prompts.py \
#    --prompt_dataset /workspace/ColossalAI/data/InstructionWild/instinwild_ch.json \
#    --pretrain_dataset /workspace/ColossalAI/data/InstructionWild/instinwild_ch.json \
#    --strategy colossalai_zero2 \
#    --pretrain /workspace/ColossalAI/Saved/Coati-7B-sft \
#    --model 'llama' \
#    --rm_pretrain /workspace/ColossalAI/Saved/Coati-7B-sft \
#    --rm_path /workspace/ColossalAI/Saved/Coati-7B-rw.pt
