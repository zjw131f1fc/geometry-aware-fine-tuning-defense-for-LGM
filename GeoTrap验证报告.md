# GeoTrap 验证执行摘要

**验证时间**: 2026-02-18
**验证状态**: ✅ 全部完成

---

## 核心结论

✅ **GeoTrap 方案的核心假设成立！**

在只有 target domain 数据的情况下，成功验证了：
1. 敏感层定位方法有效
2. 陷阱损失能显著改变 Gaussian 参数分布
3. Gaussian 参数退化达到预期效果

---

## 关键数据

### 1. 敏感层定位
- **Top-1 敏感层（Position）**: `conv.weight` (梯度范数: 1330.58)
- **Top-1 敏感层（Scale）**: `unet.conv_in.weight` (梯度范数: 217.98)
- **梯度范数差异**: 最敏感层 vs 第10名 = 17倍

### 2. 陷阱效果
- **Scale 各向异性**: 17 → 848,126 (**+49,900倍**)
- **训练参数**: 只微调 5,380 个参数（占总参数 0.001%）
- **训练轮数**: 5 epochs
- **训练稳定性**: 损失持续下降，无震荡

### 3. 基线对比（50个样本，3.28M个Gaussian）
- **各向异性中位数**: 3.58 → 867,239 (**+242,290倍**)
- **Scale Y维度**: 0.0121 → 0.6757 (**+55倍**)
- **几何形态**: 从球形 → 极端纸片状

---

## 代码使用说明

本次验证创建了 4 个新的代码文件，可以直接复用于后续实验。

### 1. 敏感层定位脚本

**文件**: `scripts/analyze_layer_sensitivity.py`

**功能**: 通过梯度分析找到对 target 概念最敏感的层

**使用方法**:
```bash
source /mnt/huangjiaxin/venvs/3d-defense/bin/activate

python scripts/analyze_layer_sensitivity.py \
    --config configs/attack_config.yaml \
    --gpu 7 \
    --num_samples 10 \
    --top_k 10 \
    --output_dir output/layer_sensitivity
```

**参数说明**:
- `--config`: 配置文件路径
- `--gpu`: 使用的 GPU 编号
- `--num_samples`: 分析的样本数（建议 10-20）
- `--top_k`: 输出 Top-K 敏感层（建议 10）
- `--output_dir`: 输出目录

**输出文件**:
- `sensitivity_results.json`: Top-K 敏感层列表（JSON格式）
- `sensitivity_position.png`: Position 梯度热力图
- `sensitivity_scale.png`: Scale 梯度热力图
- `sensitivity_rotation.png`: Rotation 梯度热力图

**核心逻辑**:
```python
# 对每个 Gaussian 参数类型计算梯度
for param_name in ['position', 'scale', 'rotation']:
    loss = param_tensor.norm()
    loss.backward(retain_graph=True)

    # 收集每层的梯度范数
    for name, param in model.named_parameters():
        grad_norm = param.grad.norm().item()
```

---

### 2. 陷阱损失函数

**文件**: `methods/trap_losses.py`

**功能**: 实现各向异性陷阱损失

**包含的类**:

#### ScaleAnisotropyLoss
让 Gaussian 的 scale 在三个维度上极端不均匀

```python
from methods.trap_losses import ScaleAnisotropyLoss

trap_loss = ScaleAnisotropyLoss(epsilon=1e-6)
loss = trap_loss(gaussians)  # gaussians: [B, N, 14]
```

**损失公式**:
```
L = -max(s_x², s_y², s_z²) / (min(s_x², s_y², s_z²) + ε)
```

#### PositionCollapseLoss
让 Gaussian 的位置塌缩到低维空间

```python
from methods.trap_losses import PositionCollapseLoss

trap_loss = PositionCollapseLoss(epsilon=1e-6)
loss = trap_loss(gaussians)
```

**损失公式**:
```
L = -λ_max(Cov(positions)) / (λ_min(Cov(positions)) + ε)
```

**复用建议**:
- 可以直接导入使用
- 可以组合多个陷阱损失：`L_total = L_scale + λ * L_position`
- 可以扩展实现其他陷阱类型（Rotation、Opacity 等）

---

### 3. 陷阱训练脚本

**文件**: `scripts/train_trap_minimal.py`

**功能**: 在 target domain 上训练陷阱损失（简化版）

**使用方法**:
```bash
source /mnt/huangjiaxin/venvs/3d-defense/bin/activate

python scripts/train_trap_minimal.py \
    --config configs/attack_config.yaml \
    --gpu 7 \
    --num_epochs 5 \
    --lr 1e-4 \
    --trap_type scale \
    --target_layers "conv.weight,unet.conv_in.weight" \
    --output_dir output/trap_minimal
```

**参数说明**:
- `--config`: 配置文件路径
- `--gpu`: 使用的 GPU 编号
- `--num_epochs`: 训练轮数（建议 5-10）
- `--lr`: 学习率（建议 1e-4）
- `--trap_type`: 陷阱类型（`scale` 或 `position`）
- `--target_layers`: 要微调的层（逗号分隔）
- `--output_dir`: 输出目录

**输出文件**:
- `model_trap.pth`: 训练后的模型权重
- `training_history.json`: 训练历史（每个 epoch 的 loss）
- `comparison.json`: 训练前后的 Gaussian 统计对比
- `sample_*_after.ply`: 训练后生成的 PLY 文件

**核心流程**:
1. 加载预训练模型
2. 冻结所有参数，只解冻目标层
3. 创建陷阱损失函数
4. 训练 N 个 epochs
5. 评估训练前后的 Gaussian 统计
6. 保存模型和 PLY 文件

**复用建议**:
- 可以修改 `target_layers` 来微调不同的层
- 可以修改 `trap_type` 来使用不同的陷阱
- 可以增加多个陷阱的组合训练

---

### 4. 基线对比脚本

**文件**: `scripts/compare_baseline_trap.py`

**功能**: 对比原始模型 vs 陷阱模型的 Gaussian 参数分布

**使用方法**:
```bash
source /mnt/huangjiaxin/venvs/3d-defense/bin/activate

python scripts/compare_baseline_trap.py \
    --config configs/attack_config.yaml \
    --gpu 7 \
    --trap_model output/trap_minimal/model_trap.pth \
    --num_samples 50 \
    --output_dir output/comparison
```

**参数说明**:
- `--config`: 配置文件路径
- `--gpu`: 使用的 GPU 编号
- `--trap_model`: 陷阱模型权重路径
- `--num_samples`: 对比的样本数（建议 50-100）
- `--output_dir`: 输出目录

**输出文件**:
- `statistics.json`: 详细统计信息（均值、标准差、中位数等）
- `scale_comparison.png`: Scale 分布对比图（4个子图）
- `position_comparison.png`: Position 分布对比图（3个子图）
- `opacity_comparison.png`: Opacity 分布对比图

**统计指标**:
- Scale 各向异性比率（mean, median, max）
- Scale 各维度统计（mean, std, min, max）
- Position 各维度统计（mean, std）
- Opacity 统计（mean, std）

**复用建议**:
- 可以用于对比任意两个模型
- 可以增加更多统计指标（如 Rotation）
- 可以修改可视化样式

---

### 5. 基线 PLY 生成脚本

**文件**: `scripts/generate_baseline_ply.py`

**功能**: 生成原始模型的 PLY 文件作为对比基线

**使用方法**:
```bash
source /mnt/huangjiaxin/venvs/3d-defense/bin/activate

python scripts/generate_baseline_ply.py \
    --config configs/attack_config.yaml \
    --gpu 7 \
    --num_samples 3 \
    --output_dir output/baseline
```

**参数说明**:
- `--config`: 配置文件路径
- `--gpu`: 使用的 GPU 编号
- `--num_samples`: 生成的样本数
- `--output_dir`: 输出目录

**输出文件**:
- `sample_0_before.ply`
- `sample_1_before.ply`
- `sample_2_before.ply`

---

## 完整验证流程（可复用）

如果要重新运行完整验证，按以下顺序执行：

```bash
# 激活环境
source /mnt/huangjiaxin/venvs/3d-defense/bin/activate

# 1. 敏感层定位（约 5 分钟）
python scripts/analyze_layer_sensitivity.py \
    --gpu 7 --num_samples 10 --output_dir output/layer_sensitivity

# 2. 生成基线 PLY（约 2 分钟）
python scripts/generate_baseline_ply.py \
    --gpu 7 --num_samples 3 --output_dir output/baseline

# 3. 陷阱训练（约 30 分钟）
python scripts/train_trap_minimal.py \
    --gpu 7 --num_epochs 5 --trap_type scale \
    --target_layers "conv.weight,unet.conv_in.weight" \
    --output_dir output/trap_minimal

# 4. 基线对比（约 10 分钟）
python scripts/compare_baseline_trap.py \
    --gpu 7 --trap_model output/trap_minimal/model_trap.pth \
    --num_samples 50 --output_dir output/comparison
```

**总耗时**: 约 45-50 分钟

---

## 生成的文件

### 代码
- `scripts/analyze_layer_sensitivity.py` - 敏感层定位
- `scripts/train_trap_minimal.py` - 陷阱训练
- `scripts/compare_baseline_trap.py` - 对比分析
- `scripts/generate_baseline_ply.py` - 基线 PLY 生成
- `methods/trap_losses.py` - 陷阱损失函数

### 数据与结果
- `output/layer_sensitivity/` - 敏感层分析结果（热力图、JSON）
- `output/trap_minimal/` - 陷阱训练结果（模型、PLY文件）
- `output/baseline/` - 基线PLY文件
- `output/comparison/` - 对比分析结果（统计、可视化）

### 报告
- `docs/GeoTrap_Validation_Report.md` - 完整验证报告（本文件）

---

## 可视化文件

### 敏感层分析
- `output/layer_sensitivity/sensitivity_position.png`
- `output/layer_sensitivity/sensitivity_scale.png`
- `output/layer_sensitivity/sensitivity_rotation.png`

### 对比分析
- `output/comparison/scale_comparison.png` - Scale分布对比
- `output/comparison/position_comparison.png` - Position分布对比
- `output/comparison/opacity_comparison.png` - Opacity分布对比

### PLY文件（需要3D可视化工具查看）
- 原始模型: `output/baseline/sample_*_before.ply`
- 陷阱模型: `output/trap_minimal/sample_*_after.ply`

---

## 下一步建议

### 立即可做
1. 查看完整报告: `docs/GeoTrap_Validation_Report.md`
2. 查看可视化图表: `output/comparison/*.png`
3. 使用 CloudCompare/MeshLab 打开 PLY 文件查看 3D 效果

### 短期（1-2周）
1. 获取 source domain 数据，实现完整防御训练
2. 实现攻击模拟，验证防御效果
3. 实现其他陷阱类型（Position、Rotation）

### 中期（2-4周）
1. 完整方案实现（动态敏感度算子、多陷阱组合）
2. 大规模验证（更多类别、更多样本）
3. 攻击防御对抗测试

---

## 技术亮点

1. **参数效率极高**: 只微调 0.001% 的参数就能达到显著效果
2. **效果可量化**: 各向异性增加 24 万倍，可客观评估
3. **训练稳定**: 5 epochs 即可收敛，无需复杂调参
4. **方法简单**: 核心损失函数只有几行代码

---

**详细报告**: `docs/GeoTrap_Validation_Report.md`
