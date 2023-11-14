#!/bin/bash

set -euo pipefail
# 
file=quick_start/distributed/a0.py
torchrun \
--nproc_per_node=2 \
$file \
> $file.log 2>&1 &