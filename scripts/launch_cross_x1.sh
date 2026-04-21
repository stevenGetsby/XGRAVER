#!/usr/bin/env bash
# Launch cross-x1 cascade training on 8 GPUs via pixi + accelerate config file.
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"

REPO=/home/ubuntu/yizhao/3D/GRAVER_MICROSOFT
PIXI_ENV=/home/ubuntu/yizhao/3D/env
OUTPUT_DIR="${REPO}/ckpt/cross_x1_full"
ACCEL_CFG="${REPO}/configs/accelerate/8gpu_fp16.yaml"
TRAIN_CFG="${REPO}/configs/flow_matching/block_feats.json"

mkdir -p "${OUTPUT_DIR}"
cd "${PIXI_ENV}"

exec pixi run -- bash -c "
  cd ${REPO} && \
  accelerate launch \
    --config_file ${ACCEL_CFG} \
    train_accelerate.py \
    --config ${TRAIN_CFG} \
    --output_dir ${OUTPUT_DIR} \
    --mixed_precision fp16
"

