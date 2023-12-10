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

set_n_least_used_CUDA_VISIBLE_DEVICES 3

torchrun --standalone --nproc_per_node=3 /workspace/yk_repo/ColossalAI/applications/Chat/examples/train_sft.py \
    --pretrain "/share/hf_model/bloomz-560m" \
    --model 'bloom' \
    --strategy colossalai_zero2 \
    --save_path Coati-bloom-560m-sft \
    --dataset /share/hf_model/instinwild_ch.json \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1


torchrun --standalone --nproc_per_node=3 train_sft.py \
    --pretrain "/share/hf_model/llama-7b-hf" \
    --model 'llama' \
    --strategy colossalai_zero2_cpu \
    --log_interval 10 \
    --save_path  /workspace/ColossalAI/Saved/Coati-llama-7b-sft \
    --dataset /share/hf_model/instinwild_ch.json \
    --batch_size 1 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 50 \
    --max_epochs 1 \