#!/bin/bash
set -x
ulimit -c 0


# check if NUM_GPU is unset, if so set it to 8
NUM_GPU=${NUM_GPU:-8}
# SLURM_JOB_NUM_NODES to 1
SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
# MASTER_ADDR to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}
# MASTER_PORT to 29500
MASTER_PORT=${MASTER_PORT:-29500}
script_name=hma_soft_d256_gpu_${NUM_GPU}_nodes_${SLURM_JOB_NUM_NODES}

batch_size=1
gradient_accumulation_steps=1
script_name="${script_name}_16g"



torchrun   --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${NUM_GPU} \
    --rdzv-id=${SLURM_JOB_ID} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
     hma/train_multi.py --model_type continuous \
     --genie_config hma/configs/mar_n32_h8_d256_action.json \
    --output_dir data/$script_name \
    --num_episodes_per_dataset 1000000 \
    --per_device_train_batch_size $batch_size \
    --run_name $script_name \
    --resume_from_checkpoint data/$script_name/  \
    --train_split experiments/datasplit/dataset30.yaml \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --vis_every_n_steps 1000 \
    --max_grad_norm 10.0 \
    --attention_dropout 0.01 \
    --cleanup_checkpoints \
    --per_device_eval_batch_size 1 \
    --num_workers 24 \
