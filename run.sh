#!/bin/bash

set -euo pipefail
# 
file=train_qlora.py
nohup python $file \
    --train_args_file train_args/qlora/qwen-7b-sft-qlora2.json
    > $file.log 2>&1 &