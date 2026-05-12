#!/usr/bin/env bash
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/cfs/yizhao/SKET}"
OUT_DIR="${OUT_DIR:-export_modelscope_sket_feats_only_200}"
FEATS_WEIGHT="${FEATS_WEIGHT:-pretrained/modelscope_graver/feats_model_latest.pt}"
FEATS_CONFIG="${FEATS_CONFIG:-pretrained/modelscope_graver/feats_model_latest.json}"
NUM_CASES="${NUM_CASES:-200}"
MAX_BLOCK_NUM="${MAX_BLOCK_NUM:-15000}"
STEPS="${STEPS:-30}"
SEED="${SEED:-20260506}"
GPUS="${GPUS:-0,1,2,3}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
WORLD_SIZE="${#GPU_LIST[@]}"

mkdir -p "$OUT_DIR/logs"
rm -f "$OUT_DIR/DONE" "$OUT_DIR/FAILED" "$OUT_DIR/exit_code.txt"

{
    echo "DATA_ROOT=$DATA_ROOT"
    echo "OUT_DIR=$OUT_DIR"
    echo "NUM_CASES=$NUM_CASES"
    echo "MAX_BLOCK_NUM=$MAX_BLOCK_NUM"
    echo "STEPS=$STEPS"
    echo "SEED=$SEED"
    echo "GPUS=$GPUS"
    echo "WORLD_SIZE=$WORLD_SIZE"
} | tee "$OUT_DIR/logs/launcher.log"

pids=()
for rank in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$rank]}"
    log_path="$OUT_DIR/logs/rank${rank}_gpu${gpu}.log"
    echo "[launch] rank=$rank gpu=$gpu log=$log_path" | tee -a "$OUT_DIR/logs/launcher.log"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/export_feats_only_cases.py \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUT_DIR" \
        --feats_weight "$FEATS_WEIGHT" \
        --feats_config "$FEATS_CONFIG" \
        --num_cases "$NUM_CASES" \
        --max_block_num "$MAX_BLOCK_NUM" \
        --steps "$STEPS" \
        --seed "$SEED" \
        --rank "$rank" \
        --world_size "$WORLD_SIZE" \
        --device cuda \
        > "$log_path" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "$pid" > "$OUT_DIR/logs/rank${rank}.pid"
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

python - "$OUT_DIR" <<'PY'
import glob
import json
import os
import sys

import pandas as pd

out_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(out_dir, "manifest_rank*.tsv")))
frames = []
for path in paths:
    if os.path.getsize(path) == 0:
        continue
    frames.append(pd.read_csv(path, sep="\t"))

if frames:
    manifest = pd.concat(frames, ignore_index=True)
    manifest["seq"] = manifest["seq"].astype(int)
    manifest = manifest.sort_values("seq").reset_index(drop=True)
else:
    manifest = pd.DataFrame()

manifest_path = os.path.join(out_dir, "manifest.tsv")
manifest.to_csv(manifest_path, sep="\t", index=False)

status_counts = {}
if not manifest.empty and "status" in manifest:
    status_counts = manifest["status"].value_counts().to_dict()

summary = {
    "manifest_parts": paths,
    "num_rows": int(len(manifest)),
    "status_counts": {str(k): int(v) for k, v in status_counts.items()},
    "num_cases_dirs": len(glob.glob(os.path.join(out_dir, "cases", "*"))),
}
with open(os.path.join(out_dir, "summary.json"), "w") as file:
    json.dump(summary, file, indent=2)
print(json.dumps(summary, indent=2))
PY

echo "$status" > "$OUT_DIR/exit_code.txt"
if [[ "$status" == "0" ]]; then
    touch "$OUT_DIR/DONE"
else
    touch "$OUT_DIR/FAILED"
fi
exit "$status"