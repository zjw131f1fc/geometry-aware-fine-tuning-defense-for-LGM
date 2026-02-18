#!/bin/bash
# =============================================================================
# 防御超参数搜索脚本
#
# 功能：
# 1. 枚举 4 种 trap loss 的两两组合（C(4,2)=6）
# 2. 对每种组合测试多组超参数
# 3. 在可用 GPU 上并行启动（每个任务占一张卡）
# 4. 汇总所有实验结果
#
# 使用方法:
#   ./script/sweep_defense.sh <GPU_IDS>
# 示例:
#   ./script/sweep_defense.sh 4,5,6,7
#   ./script/sweep_defense.sh 0,1
# =============================================================================

set -e

if [ $# -lt 1 ]; then
    echo "使用方法: $0 <GPU_IDS>"
    echo "示例: $0 4,5,6,7"
    exit 1
fi

# ===== 配置 =====
BASE_CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="output/sweep_${TIMESTAMP}"
LOG_DIR="${SWEEP_DIR}/logs"
CONFIG_DIR="${SWEEP_DIR}/configs"
mkdir -p "$LOG_DIR" "$CONFIG_DIR"

# 解析 GPU 列表
IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}

# 设置 PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/LGM:${PYTHONPATH}"

echo "=========================================="
echo "防御超参数搜索"
echo "=========================================="
echo "GPU: ${GPUS[*]} (${NUM_GPUS} 张)"
echo "输出目录: ${SWEEP_DIR}"
echo "=========================================="

# ===== 实验网格 =====
# 损失组合：4 种 trap 的两两组合
LOSS_PAIRS=(
    "position,scale"
    "position,opacity"
    "position,rotation"
    "scale,opacity"
    "scale,rotation"
    "opacity,rotation"
)

# 超参数网格（可自定义）
LAMBDA_TRAPS=("1.0")
COUPLING_MODES=("mult_gc")
# 只遍历物理属性组合，其他参数用配置文件默认值
# mult_gc   = 乘法耦合 + 梯度冲突（默认）
# mult_only = 仅乘法耦合
# none      = 无耦合（加法组合）

# ===== 生成配置文件的 Python 脚本 =====
generate_config() {
    local pair=$1
    local lambda_trap=$2
    local coupling=$3
    local output_path=$4
    local exp_output_dir=$5

    python3 -c "
import yaml, sys

pair = '${pair}'.split(',')
lambda_trap = float('${lambda_trap}')
coupling = '${coupling}'
output_path = '${output_path}'
exp_output_dir = '${exp_output_dir}'

with open('${BASE_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)

# 设置 trap losses：只启用选中的两种
for loss_name in ['position', 'scale', 'opacity', 'rotation']:
    enabled = loss_name in pair
    config['defense']['trap_losses'][loss_name]['static'] = enabled
    config['defense']['trap_losses'][loss_name]['dynamic'] = False

# 设置 lambda
config['defense']['lambda_trap'] = lambda_trap

# 设置耦合模式
if coupling == 'mult_gc':
    config['defense']['coupling']['multiplicative'] = True
    config['defense']['coupling']['gradient_conflict']['enabled'] = True
elif coupling == 'mult_only':
    config['defense']['coupling']['multiplicative'] = True
    config['defense']['coupling']['gradient_conflict']['enabled'] = False
else:  # none
    config['defense']['coupling']['multiplicative'] = False
    config['defense']['coupling']['gradient_conflict']['enabled'] = False

with open(output_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
"
}

# ===== GPU 任务队列 =====
declare -A GPU_PIDS  # GPU_ID -> PID

wait_for_gpu() {
    # 等待任意一个 GPU 空闲，返回空闲 GPU ID
    while true; do
        for gpu in "${GPUS[@]}"; do
            pid=${GPU_PIDS[$gpu]:-""}
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                if [ -n "$pid" ]; then
                    wait "$pid" 2>/dev/null
                    local exit_code=$?
                    if [ $exit_code -eq 0 ]; then
                        echo "[GPU $gpu] 任务完成 ✓" >&2
                    else
                        echo "[GPU $gpu] 任务失败 ✗ (exit=$exit_code)" >&2
                    fi
                fi
                GPU_PIDS[$gpu]=""
                echo "$gpu"
                return
            fi
        done
        sleep 5
    done
}

# ===== 生成并启动所有实验 =====
EXP_COUNT=0
TOTAL_EXPS=0

for pair in "${LOSS_PAIRS[@]}"; do
    for lt in "${LAMBDA_TRAPS[@]}"; do
        for cm in "${COUPLING_MODES[@]}"; do
            TOTAL_EXPS=$((TOTAL_EXPS + 1))
        done
    done
done

echo ""
echo "总实验数: ${TOTAL_EXPS}"
echo "开始启动..."
echo ""

for pair in "${LOSS_PAIRS[@]}"; do
    for lt in "${LAMBDA_TRAPS[@]}"; do
        for cm in "${COUPLING_MODES[@]}"; do
            EXP_COUNT=$((EXP_COUNT + 1))
            EXP_NAME="${pair//,/_}_lt${lt}_${cm}"
            EXP_OUTPUT="${SWEEP_DIR}/${EXP_NAME}"
            EXP_CONFIG="${CONFIG_DIR}/${EXP_NAME}.yaml"

            # 生成配置
            generate_config "$pair" "$lt" "$cm" "$EXP_CONFIG" "$EXP_OUTPUT"

            # 等待空闲 GPU
            FREE_GPU=$(wait_for_gpu)

            echo "[${EXP_COUNT}/${TOTAL_EXPS}] 启动: ${EXP_NAME} → GPU ${FREE_GPU}"

            # 启动实验（后台运行）
            python script/train_defense_with_eval.py \
                --config "$EXP_CONFIG" \
                --gpu "$FREE_GPU" \
                --output_dir "$EXP_OUTPUT" \
                > "${LOG_DIR}/${EXP_NAME}.log" 2>&1 &

            GPU_PIDS[$FREE_GPU]=$!
        done
    done
done

# 等待所有剩余任务完成
echo ""
echo "所有实验已启动，等待剩余任务完成..."
wait
echo "全部完成。"

# ===== 汇总结果 =====
echo ""
echo "=========================================="
echo "汇总结果"
echo "=========================================="

python3 -c "
import json, os, glob

sweep_dir = '${SWEEP_DIR}'
results = []

for exp_dir in sorted(glob.glob(os.path.join(sweep_dir, '*/eval_summary.json'))):
    exp_name = os.path.basename(os.path.dirname(exp_dir))
    with open(exp_dir) as f:
        summary = json.load(f)
    baseline = next((r for r in summary if r['tag'] == 'baseline'), None)
    final = next((r for r in summary if r['tag'] == 'final'), None)
    if baseline and final:
        results.append({
            'experiment': exp_name,
            'baseline_psnr': baseline['attack_psnr'],
            'final_psnr': final['attack_psnr'],
            'delta_psnr': final['attack_psnr'] - baseline['attack_psnr'],
            'baseline_lpips': baseline['attack_lpips'],
            'final_lpips': final['attack_lpips'],
            'delta_lpips': final['attack_lpips'] - baseline['attack_lpips'],
        })

if not results:
    print('没有找到完成的实验结果')
else:
    # 按 delta_lpips 排序（越正越好 = 攻击感知质量越差 = 防御越有效）
    results.sort(key=lambda r: -r['delta_lpips'])

    print(f\"{'实验':<35} {'Base LPIPS':>11} {'Final LPIPS':>12} {'ΔLPIPS':>8} {'ΔPSNR':>8}\")
    print('-' * 80)
    for r in results:
        print(f\"{r['experiment']:<35} {r['baseline_lpips']:>11.4f} {r['final_lpips']:>12.4f} {r['delta_lpips']:>+8.4f} {r['delta_psnr']:>+8.2f}\")
    print('-' * 80)
    best = results[0]
    print(f\"\\n最佳: {best['experiment']} (ΔLPIPS={best['delta_lpips']:+.4f})\")

    # 保存汇总 JSON
    out_path = os.path.join(sweep_dir, 'sweep_summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'汇总保存: {out_path}')
"

echo ""
echo "日志目录: ${LOG_DIR}"
echo "输出目录: ${SWEEP_DIR}"
