#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

python3 main.py --config-path hf_configs/llama3-8B_config.json  \
  --device-type A100 \
  --num-nodes 1 \
  --world-size 4 \
  --tp-size 4 \
  --target-isl 2048 \
  --target-osl 2048 \
  --batch-size 4
