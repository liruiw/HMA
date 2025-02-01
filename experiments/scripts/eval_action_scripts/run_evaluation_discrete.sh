#!/bin/bash

set -x
ulimit -c 0

script_name=${1}
dataset=${2}

# evaluate
DATASET=${script_name}_${dataset}_valset
mkdir -p data/${DATASET}/output

MASTER_PORT=$(python -c ‘import socket; s=socket.socket(); s.bind((“”, 0)); print(s.getsockname()[1]); s.close()’)
export MASTER_PORT="${MASTER_PORT:-29555 }"

accelerate launch --main_process_port $MASTER_PORT hma/evaluate.py --checkpoint_dir data/${script_name}  \
    --val_data_dir data/${dataset}_magvit_traj1000000_val   --save_outputs_dir data/${DATASET}

python hma/generate.py   --checkpoint_dir data/${script_name}  \
    --val_data_dir data/${dataset}_magvit_traj1000000_val \
    --output_dir data/${DATASET}/output

# visualize
python hma/visualize.py --token_dir data/${DATASET}/output
