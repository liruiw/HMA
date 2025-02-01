#!/bin/bash
set -x
ulimit -c 0



source venv/bin/activate


# mv the data folder

# python -m hma.train_multi --genie_config hma/configs/magvit_n32_h8_d256_action.json \
#     --output_dir data/$script_name \
#     --max_eval_steps 10 \
#     --num_episodes_per_dataset 1000000 \
#     --per_device_train_batch_size 1 \
#     --train_split experiments/datasplit/dataset1.yaml

python -m hma.train_multi_diffusion --genie_config hma/configs/mar_n32_h8_d256_action.json \
    --output_dir data/$script_name \
    --max_eval_steps 10 \
    --num_episodes_per_dataset 1000000 \
     --model_type continuous \
    --per_device_train_batch_size 1 \
    --train_split experiments/datasplit/dataset1.yaml

