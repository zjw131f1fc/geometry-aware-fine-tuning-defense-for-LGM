# GeoTrap 防御实验记录

## 实验概述

基于 LGM (Large Multi-View Gaussian Model) 的 GeoTrap 防御方法实验。通过在模型权重中嵌入几何陷阱（position collapse、scale anisotropy、opacity collapse），使攻击者（LoRA 微调）无法正确重建 target 类别的 3D 物体。

## 实验一：基线防御（10 epochs, 2 traps, 3 层）

### 配置

| 参数 | 值 |
|------|-----|
| defense_epochs | 10 |
| lr | 0.00005 |
| batch_size | 1 |
| gradient_accumulation_steps | 4 (有效 batch_size=4) |
| weight_decay | 0.05 |
| gradient_clip | 1.0 |
| lambda_trap | 1.0 |
| lambda_distill | 1.0 |
| source_ratio | 0.8 |
| source.max_samples | 200 (objaverse) |
| target categories | knife, hammer, scissor |
| target.max_samples_per_category | 5 |
| target.object_offset | 15 |
| target.samples_per_object | 3 |

**Trap 配置：**
- position: static=true, dynamic=false
- scale: static=true, dynamic=false
- opacity: static=false
- rotation: static=false

**耦合配置：**
- multiplicative: true
- gradient_conflict: enabled=true, weight=0.1, every_k_steps=10

**防御层（3 组 ResBlock）：**
- unet.down_blocks.0.nets.0
- unet.down_blocks.0.nets.1
- unet.down_blocks.1.nets.0

### 结果

| Epoch | position_static | scale_static | source_distill_mse | coupling_value | grad_cosine_sim |
|-------|----------------|-------------|-------------------|---------------|----------------|
| 1 | -1.44 | -2.73 | 0.00128 | -8.09 | -0.159 |
| 2 | -1.68 | -3.26 | 0.00147 | -10.41 | -0.441 |
| 3 | -1.98 | -3.73 | 0.00225 | -13.12 | +0.195 |
| 4 | -2.16 | -4.16 | 0.00288 | -15.32 | +0.131 |
| 5 | -2.43 | -4.21 | 0.00247 | -16.89 | -0.343 |
| 6 | -2.75 | -4.48 | 0.00317 | -19.59 | -0.247 |
| 7 | -3.54 | -4.92 | 0.00499 | -25.87 | -0.028 |
| 8 | -4.03 | -5.36 | 0.00672 | -30.96 | -0.531 |
| 9 | -4.33 | -5.54 | 0.00617 | -33.85 | -0.154 |
| 10 | -4.50 | -5.65 | 0.00613 | -35.57 | -0.126 |

### 评价

- Trap 单调增强但 E10 未收敛
- source_distill_mse 增长 4.8x（0.00128 → 0.00613），E9-E10 企稳
- 防御效果不足，攻击后 target 类别仍可正常重建

---

## 实验二：增强防御（25 epochs, 2 traps, 6 层）✅ 防御成功

### 相对实验一的改动

1. defense_epochs: 10 → 25
2. 防御层从 3 组扩到 6 组（新增 Tier-1 层）

### 配置

| 参数 | 值 |
|------|-----|
| defense_epochs | 25 |
| lr | 0.00005 |
| batch_size | 1 |
| gradient_accumulation_steps | 4 (有效 batch_size=4) |
| weight_decay | 0.05 |
| gradient_clip | 1.0 |
| lambda_trap | 1.0 |
| lambda_distill | 1.0 |
| source_ratio | 0.8 |
| source.max_samples | 200 (objaverse) |
| target categories | knife, hammer, scissor |
| target.max_samples_per_category | 5 |
| target.object_offset | 15 |
| target.samples_per_object | 3 |

**Trap 配置：**
- position: static=true, dynamic=false
- scale: static=true, dynamic=false
- opacity: static=false
- rotation: static=false

**耦合配置：**
- multiplicative: true（两项乘法耦合）
- gradient_conflict: enabled=true, weight=0.1, every_k_steps=10

**防御层（6 组 ResBlock）：**
- unet.down_blocks.0.nets.0 (核心层, ratio ~1.7)
- unet.down_blocks.0.nets.1 (核心层, ratio ~1.7)
- unet.down_blocks.1.nets.0 (核心层, ratio ~1.5)
- unet.down_blocks.1.nets.1 (Tier-1, ratio ~1.3)
- unet.down_blocks.2.nets.0 (Tier-1, ratio ~1.2)
- unet.down_blocks.2.nets.1 (Tier-1, ratio ~1.2)

### 结果

| Epoch | position_static | scale_static | source_distill_mse | coupling_value | grad_cosine_sim |
|-------|----------------|-------------|-------------------|---------------|----------------|
| 1 | -2.03 | -4.67 | 0.00268 | -16.19 | -0.195 |
| 2 | -4.26 | -6.30 | 0.00823 | -37.41 | -0.340 |
| 3 | -5.48 | -6.82 | 0.00787 | -49.65 | +0.043 |
| 4 | -6.20 | -7.04 | 0.00768 | -56.90 | +0.087 |
| 5 | -6.45 | -7.25 | 0.00690 | -60.47 | -0.164 |
| 6 | -6.62 | -7.48 | 0.00648 | -63.69 | +0.298 |
| 7 | -6.92 | -7.53 | 0.00562 | -66.56 | +0.046 |
| 8 | -6.17 | -7.83 | 0.00688 | -62.32 | -0.179 |
| 9 | -7.07 | -7.76 | 0.00517 | -69.71 | -0.064 |
| 10 | -7.06 | -7.87 | 0.00549 | -70.55 | -0.158 |
| 11 | -7.63 | -8.07 | 0.00582 | -77.25 | -0.197 |
| 12 | -7.60 | -8.09 | 0.00519 | -77.10 | -0.010 |
| 13 | -7.05 | -8.28 | 0.00570 | -73.67 | +0.003 |
| 14 | -7.68 | -8.33 | 0.00511 | -80.04 | -0.059 |
| 15 | -6.82 | -8.25 | 0.00475 | -71.35 | -0.020 |
| 16 | -7.40 | -8.37 | 0.00528 | -77.71 | -0.139 |
| 17 | -7.75 | -8.57 | 0.00541 | -82.70 | -0.172 |
| 18 | -7.77 | -8.47 | 0.00436 | -82.07 | +0.128 |
| 19 | -8.25 | -8.64 | 0.00468 | -88.22 | -0.017 |
| 20 | -8.37 | -8.71 | 0.00444 | -90.04 | -0.141 |
| 21 | -8.38 | -8.74 | 0.00469 | -90.33 | -0.038 |
| 22 | -8.48 | -8.84 | 0.00480 | -92.23 | -0.184 |
| 23 | -8.13 | -8.90 | 0.00438 | -89.33 | -0.100 |
| 24 | -8.19 | -8.79 | 0.00420 | -88.98 | +0.037 |
| 25 | -7.30 | -8.97 | 0.00467 | -81.77 | -0.189 |

### 实验一 vs 实验二对比

| 指标 | 实验一 (E10) | 实验二 (E25) | 变化 |
|------|-------------|-------------|------|
| position_static | -4.50 | -7.30 (峰值 -8.48) | 1.6x |
| scale_static | -5.65 | -8.97 | 1.6x |
| coupling_value | -35.57 | -81.77 (峰值 -92.23) | 2.3x |
| source_distill_mse | 0.00613 | 0.00467 | 改善 24% |

### 攻击测试结果

使用 LoRA 攻击（r=8, alpha=16, target_modules=[qkv, proj]）对防御后模型进行微调：
- PSNR 数值看似正常
- 实际渲染图像严重模糊、扭曲，target 类别物体不可辨认
- **防御成功**：攻击者无法通过 LoRA 微调恢复 target 类别的正常重建能力

### 评价

- 两项改动（epoch↑、更多层）协同效果显著
- source_distill_mse 反而比实验一更低（0.00467 vs 0.00613），更多层提供了更好的 trap/distill 分离能力
- 两项乘法耦合（position × scale）比实验一强 2.3 倍（更多 epoch + 更多层的累积效果）
- position_static 在后半段有波动（-6.82 ~ -8.48），scale_static 更稳定
- grad_cosine_sim 在零附近波动，两个 trap 梯度保持大致正交

---

## 防御机制说明

### 静态陷阱损失

- **PositionCollapseLoss**: L = -mean(log(λ_max / (λ_min + ε)))，通过协方差矩阵特征值比率衡量位置分布的各向异性
- **ScaleAnisotropyLoss**: L = -mean(log(max(s²) / (min(s²) + ε)))，衡量 Gaussian 尺度的各向异性
- **OpacityCollapseLoss**: L = mean(opacity) - 1，推动透明度趋向 0

### 乘法耦合

组合公式：L_combined = -(∏(1 - L_i) - 1)

各 L_i < 0（最小化），所以 (1 - L_i) > 1，乘积随 trap 数量指数增长。梯度 ∂L/∂L_i = -∏_{j≠i}(1 - L_j)，即每个 trap 的梯度被其他 trap 的强度放大。

### 梯度冲突正则

L_conflict = mean_{i<j} ReLU(cos_sim(g_i, g_j))

惩罚不同 trap 梯度在权重空间的对齐度，强制正交，使攻击者无法用单一梯度方向同时修复多个 trap。

### 防御层选择原则

1. 高差异 ratio（target_grad / source_grad）：对 target 数据敏感
2. LoRA 安全：位于 down_blocks.0-2，无 attention 模块，LoRA 无法直接修改
3. 处理低级形状特征：在网络早期，影响所有下游计算
