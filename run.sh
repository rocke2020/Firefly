#!/bin/bash

set -euo pipefail
# 
file=train_qlora.py
torchrun \
--nproc_per_node=4 \
$file \
--train_args_file train_args/qlora/llama2-sft-qlora-chem-gene.json \
> $file.log 2>&1 &
