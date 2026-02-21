#!/bin/bash
# 测试攻击对防御模型的效果
# 用法: ./script/test_attack_on_defense.sh <GPU_ID> <defense_model_path> <trap_combo>

set -e

GPU_ID=${1:-0}
DEFENSE_MODEL=${2:-"output/sweep_combos_20260221_011903/position+scale/model_defense.pth"}
TRAP_COMBO=${3:-"position+scale"}

echo "=========================================="
echo "测试攻击对防御模型的效果"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "防御模型: $DEFENSE_MODEL"
echo "Trap 组合: $TRAP_COMBO"
echo "=========================================="

# 检查防御模型是否存在
if [ ! -f "$DEFENSE_MODEL" ]; then
    echo "错误: 防御模型不存在: $DEFENSE_MODEL"
    exit 1
fi

# 创建临时配置文件
TEMP_CONFIG="configs/temp_attack_test_${TRAP_COMBO}.yaml"
cp configs/config.yaml "$TEMP_CONFIG"

# 修改配置：指向防御模型
python -c "
import yaml
with open('$TEMP_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['model']['resume'] = '$DEFENSE_MODEL'
config['training']['attack_epochs'] = 10  # 测试用 10 epochs
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f)
print('临时配置已创建: $TEMP_CONFIG')
"

# 运行攻击
echo ""
echo "开始攻击训练..."
CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch \
    --config_file acc_configs/gpu1.yaml \
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
