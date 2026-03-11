#!/bin/bash
# Stage 2: Train per-block submask with sparse flow matching

MODE=${1:-test}

echo "=========================================="
echo "Mask Training (mode=$MODE)"
echo "  → 4³ submask sparse flow matching"
echo "=========================================="

CONFIG_FILE="./configs/flow_matching/block_mask.json"

if [ "$MODE" = "test" ]; then
    OUTPUT_DIR="../ckpt/mask_test"
    LOAD_DIR="../ckpt/mask_test"
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    CKPT="latest"
elif [ "$MODE" = "train" ]; then
    export SKIP_SNAPSHOT=1
    OUTPUT_DIR="./ckpt/mask_train"
    LOAD_DIR="./ckpt/mask_train"
    CKPT="latest"
else
    echo "错误: 未知模式 '$MODE'。请使用 'test' 或 'train'。"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ -n "$LOAD_DIR" ] && [ ! -d "$LOAD_DIR" ]; then
    echo "警告: 加载目录不存在: $LOAD_DIR，从头训练..."
    LOAD_DIR=""
fi

TRAIN_CMD="accelerate launch --config_file configs/accelerate/default_config.yaml train_accelerate.py --config $CONFIG_FILE --output_dir $OUTPUT_DIR --mixed_precision bf16"

[ -n "$LOAD_DIR" ] && TRAIN_CMD="$TRAIN_CMD --load_dir $LOAD_DIR"
[ -n "$CKPT" ] && TRAIN_CMD="$TRAIN_CMD --ckpt $CKPT"

echo "配置: $CONFIG_FILE"
echo "输出: $OUTPUT_DIR"
echo "命令: $TRAIN_CMD"
echo "=========================================="

eval $TRAIN_CMD

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "训练完成！输出: $OUTPUT_DIR"
    echo "=========================================="
else
    echo "=========================================="
    echo "训练失败！"
    echo "请检查日志文件: $OUTPUT_DIR"
    echo "=========================================="
    exit 1
fi
