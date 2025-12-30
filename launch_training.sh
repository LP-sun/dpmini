#!/bin/bash
# 快速启动脚本 - DeepMD 修复后训练

set -e

PROJECT_DIR="/home/ubuntu/pj"
CONDA_ENV="cuda_env"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DeepMD Training Launcher${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$PROJECT_DIR"

# 1. 验证环境
echo -e "\n${YELLOW}[1/4]${NC} 检查 Conda 环境..."
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo -e "${RED}❌ 环境 $CONDA_ENV 不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 环境检查通过${NC}"

# 2. 运行测试
echo -e "\n${YELLOW}[2/4]${NC} 运行健康检查 (test_training_fixes.py)..."
source activate "$CONDA_ENV"
python test_training_fixes.py > /tmp/test_health_$TIMESTAMP.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 健康检查通过${NC}"
else
    echo -e "${YELLOW}⚠️  健康检查失败，但继续${NC}"
fi

# 3. 创建日志目录
echo -e "\n${YELLOW}[3/4]${NC} 创建目录..."
mkdir -p logs checkpoints_formal_long exports_formal_long
echo -e "${GREEN}✓ 目录就绪${NC}"

# 4. 启动训练
echo -e "\n${YELLOW}[4/4]${NC} 启动训练..."
echo -e "${GREEN}命令：${NC}"
echo "python -u train_cuda_optimized.py \\"
echo "  --config config_formal_100k.json \\"
echo "  --checkpoint-dir checkpoints_formal_long \\"
echo "  --export-dir exports_formal_long \\"
echo "  --force-loss mse \\"
echo "  --grad-accumulation-steps 8 \\"
echo "  2>&1 | tee logs/train_formal_$TIMESTAMP.log &"
echo ""

python -u train_cuda_optimized.py \
  --config config_formal_100k.json \
  --checkpoint-dir checkpoints_formal_long \
  --export-dir exports_formal_long \
  --force-loss mse \
  --grad-accumulation-steps 8 \
  2>&1 | tee logs/train_formal_$TIMESTAMP.log &

TRAIN_PID=$!
echo -e "${GREEN}✓ 训练已启动 (PID: $TRAIN_PID)${NC}"
echo -e "${GREEN}✓ 日志文件: logs/train_formal_$TIMESTAMP.log${NC}"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}启动成功！${NC}"
echo -e "监控日志命令："
echo -e "  tail -f logs/train_formal_$TIMESTAMP.log"
echo -e "\n查看 f_rmse 趋势:"
echo -e "  grep 'f_rmse=' logs/train_formal_$TIMESTAMP.log | tail -20"
echo -e "${BLUE}========================================${NC}"
