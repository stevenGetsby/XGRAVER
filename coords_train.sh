#!/bin/bash
# Stage 1: Train 64³ occupancy with dense flow matching

MODE=${1:-test}
echo "=========================================="
echo "Stage 1: Coords Training (mode=$MODE)"
echo "=========================================="

CONFIG_FILE="./configs/flow_matching/block_coords.json"

if [ "$MODE" = "test" ]; then
    OUTPUT_DIR="../ckpt/coords_test"
    LOAD_DIR="../ckpt/coords_test"
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    CKPT="latest"
elif [ "$MODE" = "train" ]; then
    export SKIP_SNAPSHOT=1
    OUTPUT_DIR="./ckpt/coords_train"
    LOAD_DIR="./ckpt/coords_train"
    CKPT="latest"
else
    echo "错误: 未知模式 '$MODE'。请使用 'test' 或 'train'。"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
[ ! -f "$CONFIG_FILE" ] && echo "错误: $CONFIG_FILE 不存在" && exit 1
[ -n "$LOAD_DIR" ] && [ ! -d "$LOAD_DIR" ] && echo "警告: $LOAD_DIR 不存在，从头训练..." && LOAD_DIR=""

TRAIN_CMD="accelerate launch --config_file configs/accelerate/default_config.yaml train_accelerate.py --config $CONFIG_FILE --output_dir $OUTPUT_DIR --mixed_precision bf16"
[ -n "$LOAD_DIR" ] && TRAIN_CMD="$TRAIN_CMD --load_dir $LOAD_DIR"
[ -n "$CKPT" ] && TRAIN_CMD="$TRAIN_CMD --ckpt $CKPT"

echo "命令: $TRAIN_CMD"
echo "=========================================="
eval $TRAIN_CMD
[ $? -eq 0 ] && echo "训练完成！输出: $OUTPUT_DIR" || { echo "训练失败！"; exit 1; }
