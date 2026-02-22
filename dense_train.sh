# --- 1. 根据输入参数选择配置模式 ---
MODE=${1:-test} # 从第一个参数获取模式，如果为空，则默认为 'test'

echo "=========================================="
echo "当前运行模式: $MODE"
echo "=========================================="
CONFIG_FILE="./configs/flow_matching/dense_test.json"
# --- 2. 定义不同模式下的配置 ---
if [ "$MODE" = "test" ]; then
    # --- 测试配置 ---
    OUTPUT_DIR="../ckpt/dense_test"
    LOAD_DIR="../ckpt/dense_test"
    
    # echo "正在为test模式清空输出目录: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # DATA_DIR 可以是单个目录或多个目录（用逗号分隔）
    # 例如: DATA_DIR="/path/to/data1" 或 DATA_DIR="/path/to/data1,/path/to/data2"
    DATA_DIR="/mnt/data/yizhao/JiTData_Test"
    DATA_DIR="/mnt/data/yizhao/TRAIN_DATA"
    CKPT="latest"


elif [ "$MODE" = "train" ]; then
    # --- 正式训练配置 ---
    export SKIP_SNAPSHOT=1
    OUTPUT_DIR="./ckpt/s_flow_refine/256_64_7_13"
    LOAD_DIR="./ckpt/s_flow_refine/256_64_7_13" 
    # DATA_DIR 可以是单个目录或多个目录（用逗号分隔）
    # 例如: DATA_DIR="/path/to/data1" 或 DATA_DIR="/path/to/data1,/path/to/data2"
    DATA_DIR="/mnt/data/yizhao/TRAIN_DATA"
    CKPT="latest"
    # 如果需要从之前的训练继续，可以在这里指定 LOAD_DIR
    # LOAD_DIR="/home/ubuntu/yizhao/Code/TRAIN_LOG/ss/controlnet-imgcond"

else
    echo "错误: 未知的模式 '$MODE'。请使用 'test' 或 'train'。"
    exit 1
fi

# --- 3. 脚本主体 (这部分无需改动) ---

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 基础检查
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查数据目录（支持多个目录用逗号分隔）
IFS=',' read -ra DATA_DIRS <<< "$DATA_DIR"
for dir in "${DATA_DIRS[@]}"; do
    dir=$(echo "$dir" | xargs) # 去除前后空格
    if [ ! -d "$dir" ]; then
        echo "错误: 数据目录不存在: $dir"
        exit 1
    fi
done

# 检查加载目录
if [ -n "$LOAD_DIR" ] && [ ! -d "$LOAD_DIR" ]; then
    echo "警告: 加载目录不存在: $LOAD_DIR"
    echo "将从头开始训练..."
    LOAD_DIR=""
fi

# 构建训练命令
TRAIN_CMD="accelerate launch --config_file configs/accelerate/default_config.yaml train_accelerate.py --config $CONFIG_FILE --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --mixed_precision bf16"

if [ -n "$LOAD_DIR" ]; then
    TRAIN_CMD="$TRAIN_CMD --load_dir $LOAD_DIR"
fi

if [ -n "$CKPT" ]; then
    TRAIN_CMD="$TRAIN_CMD --ckpt $CKPT"
fi

# 显示训练信息
echo "=========================================="
echo "开始训练..."
echo "使用 GPUs: $CUDA_VISIBLE_DEVICES"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "数据目录: $DATA_DIR"
echo "GPU数量: $NUM_GPUS"
if [ -n "$LOAD_DIR" ]; then
    echo "加载目录: $LOAD_DIR"
fi
if [ -n "$CKPT" ]; then
    echo "检查点: $CKPT"
fi
echo "=========================================="

# 执行训练命令
echo "执行命令: $TRAIN_CMD"
echo "=========================================="

eval $TRAIN_CMD

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "训练完成！"
    echo "输出保存在: $OUTPUT_DIR"
    echo "=========================================="
else
    echo "=========================================="
    echo "训练失败！"
    echo "请检查日志文件: $OUTPUT_DIR"
    echo "=========================================="
    exit 1
fi