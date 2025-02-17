#!/usr/bin/bash

python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE python -m pip install flash-attn==2.5.8 --no-build-isolation
mkdir data/
wget -O data/magvit2.ckpt https://huggingface.co/datasets/1x-technologies/worldmodel/resolve/main/magvit2.ckpt
