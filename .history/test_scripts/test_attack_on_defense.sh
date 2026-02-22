#!/bin/bash
# 测试攻击对防御模型的效果
# 用法: ./script/test_attack_on_defense.sh <GPU_ID> <defense_model_path>

set -e

if [ $# -lt 2 ]; then
    echo "使用方法: $0 <GPU_ID> <defense_model_path>"
    echo "示例: $0 0 output/sweep_combos_xxx/position+scale/model_defense.pth"
    exit 1
fi

GPU_ID=$1
DEFENSE_MODEL=$2

echo "=========================================="
echo "测试攻击对防御模型的效果"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "防御模型: $DEFENSE_MODEL"
echo "=========================================="

# 检查防御模型是否存在
if [ ! -f "$DEFENSE_MODEL" ]; then
    echo "错误: 防御模型不存在: $DEFENSE_MODEL"
    exit 1
fi

# 设置项目根目录到 PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/LGM:${PYTHONPATH}"

# 创建临时配置文件
TEMP_CONFIG="configs/temp_attack_test.yaml"
cp configs/config.yaml "$TEMP_CONFIG"

# 修改配置：指向防御模型
python -c "
import yaml
with open('$TEMP_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['model']['resume'] = '$DEFENSE_MODEL'
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f)
print('临时配置已创建: $TEMP_CONFIG')
print('使用配置文件中的 attack_epochs 设置')
"

# 运行攻击
echo ""
echo "开始攻击训练..."
export CUDA_VISIBLE_DEVICES=$GPU_ID
accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    script/attack_train_ddp.py \
    --config "$TEMP_CONFIG"

echo ""
echo "=========================================="
echo "攻击完成！"
echo "查看结果:"
echo "  - 防御指标日志: output/attack_*/defense_metrics_log.json"
echo "  - 渲染结果: output/attack_*/renders/"
echo ""
echo "可视化指标变化:"
echo "  python script/plot_defense_metrics.py output/attack_*/defense_metrics_log.json"
echo "=========================================="

# 清理临时配置
rm -f "$TEMP_CONFIG"
