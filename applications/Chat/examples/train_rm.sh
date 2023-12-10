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

torchrun --standalone --nproc_per_node=3 /workspace/yk_repo/ColossalAI/applications/Chat/examples/train_reward_model.py \
   --pretrain Coati-bloom-560m-sft \
   --model 'bloom' \
   --strategy colossalai_zero2 \
   --loss_fn 'log_exp'\
   --save_path Coati-bloom-560m-rw.pt \
   --dataset 'Anthropic/hh-rlhf' \
   --test True


# 上一步都OOM，无法生成模型
#torchrun --standalone --nproc_per_node=2 /workspace/yk_repo/ColossalAI/applications/Chat/examples/ \
#   --pretrain /workspace/ColossalAI/Saved/Coati-7B-sft \
#   --model 'llama' \
#   --strategy colossalai_zero2 \
#   --loss_fn 'log_exp'\
#   --save_path Coati-7B-rw.pt \
#   --dataset '/share/hf_model/hh-rlhf' \
#   --test True
