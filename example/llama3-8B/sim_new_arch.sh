#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

python3 main.py --config-path hf_configs/llama3-8B_config.json  \
  --device-type NewArch \
  --num-nodes 1 \
  --world-size 1 \
  --tp-size 1 \
  --target-isl 2048 \
  --target-osl 2048 \
  --batch-size 1
