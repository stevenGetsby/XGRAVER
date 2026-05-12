#!/usr/bin/env bash
set -eo pipefail

cd /home/ubuntu/yizhao/3D/env
export PATH="$HOME/.pixi/bin:$PATH"
eval "$(pixi shell-hook)"
set -u

cd /home/ubuntu/yizhao/3D/GRAVER_MICROSOFT

RUN_NAME="feats_predmask_mask64dice_15000_2gpu"
MASK_DIR="/home/ubuntu/yizhao/3D/GRAVER_MICROSOFT/ckpt/mask_64_dice"
MASK_CONFIG="$MASK_DIR/config.json"
MASK_WEIGHT="$MASK_DIR/ckpts/denoiser_ema0.999_step0070000.pt"
STAGE3_CONFIG="configs/flow_matching/block_feats_predmask_15000.json"
OUT_DIR="/home/ubuntu/yizhao/3D/GRAVER_MICROSOFT/ckpt/$RUN_NAME"
WORK_DIR="/tmp/graver_predmask_stage3_15000"
ENCODE_LOG="logs/encode_mask64dice_predmask15000.log"
TRAIN_LOG="logs/${RUN_NAME}.log"
ENCODE_WORLD_SIZE="${ENCODE_WORLD_SIZE:-8}"

mkdir -p logs "$WORK_DIR" "$OUT_DIR"

python - <<'PY'
import os
import pandas as pd

COL_PREFIX = "64_15_occ8"
ROOTS = ["/mnt/data/yizhao/TRAIN", "/cfs/yizhao/SKET"]
OUT = {
    "/mnt/data/yizhao/TRAIN": "/tmp/graver_predmask_stage3_15000/train_instances.txt",
    "/cfs/yizhao/SKET": "/tmp/graver_predmask_stage3_15000/sket_instances.txt",
}
ALL_OUT = "/tmp/graver_predmask_stage3_15000/instances.tsv"

selected = []
for root in ROOTS:
    metadata = pd.read_csv(os.path.join(root, "metadata.csv"))
    metadata = metadata[metadata[f"{COL_PREFIX}_block_status"] == "success"]
    metadata = metadata[metadata[f"{COL_PREFIX}_num_blocks"] <= 13500]
    metadata = metadata[metadata["cond_rendered"].fillna(False).astype(bool)]
    for sha256 in metadata["sha256"].values:
        if len(selected) < 15000:
            selected.append((root, sha256))

if len(selected) != 15000:
    raise SystemExit(f"Expected 15000 selected samples, got {len(selected)}")

for path in OUT.values():
    open(path, "w").close()
open(ALL_OUT, "w").close()
for root, sha256 in selected:
    with open(OUT[root], "a") as f:
        f.write(f"{sha256}\n")
    with open(ALL_OUT, "a") as f:
        f.write(f"{root}\t{sha256}\n")

for root, path in OUT.items():
    with open(path) as f:
        n = sum(1 for _ in f)
    print(f"[instances] {root}: {n} -> {path}", flush=True)
print(f"[instances] total: {len(selected)} -> {ALL_OUT}", flush=True)
PY

: > "$ENCODE_LOG"

run_encode_root() {
    local root="$1"
    local instances="$2"
    echo "[encode] root=$root instances=$instances world_size=$ENCODE_WORLD_SIZE" | tee -a "$ENCODE_LOG"
    for rank in $(seq 0 $((ENCODE_WORLD_SIZE - 1))); do
        CUDA_VISIBLE_DEVICES="$rank" python -m graver.dataset_toolkits.encode_mask \
            --root "$root" \
            --instances "$instances" \
            --rank "$rank" \
            --world_size "$ENCODE_WORLD_SIZE" \
            --device cuda \
            --max_block_num 13500 \
            --mask_config "$MASK_CONFIG" \
            --mask_weight "$MASK_WEIGHT" \
            --threshold 0.4 \
            --force \
            >> "$ENCODE_LOG" 2>&1 &
    done
    wait
}

run_encode_root "/mnt/data/yizhao/TRAIN" "$WORK_DIR/train_instances.txt"
run_encode_root "/cfs/yizhao/SKET" "$WORK_DIR/sket_instances.txt"

python - <<'PY'
import os
import zipfile

COL_PREFIX = "64_15_occ8"
BLOCK_FOLDER = f"blocks_{COL_PREFIX}"
LISTS = [
    ("/mnt/data/yizhao/TRAIN", "/tmp/graver_predmask_stage3_15000/train_instances.txt"),
    ("/cfs/yizhao/SKET", "/tmp/graver_predmask_stage3_15000/sket_instances.txt"),
]

total = ok = missing = 0
for root, list_path in LISTS:
    with open(list_path) as f:
        ids = [line.strip() for line in f if line.strip()]
    for sha256 in ids:
        total += 1
        npz_path = os.path.join(root, BLOCK_FOLDER, f"{sha256}.npz")
        try:
            with zipfile.ZipFile(npz_path) as zf:
                names = set(zf.namelist())
            if "pred_mask.npy" in names and "fine_feats.npy" in names:
                ok += 1
            else:
                missing += 1
        except Exception:
            missing += 1

print(f"[verify] pred_mask+fine_feats ok={ok}/{total}, missing={missing}", flush=True)
if total != 15000 or ok != 15000:
    raise SystemExit("pred_mask verification failed")
PY

accelerate launch \
    --config_file configs/accelerate/2gpu_bf16_01.yaml \
    train_accelerate.py \
    --config "$STAGE3_CONFIG" \
    --output_dir "$OUT_DIR" \
    --mixed_precision bf16 \
    2>&1 | tee "$TRAIN_LOG"