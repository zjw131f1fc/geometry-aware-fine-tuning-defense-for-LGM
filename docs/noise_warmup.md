# 噪声Warmup功能说明

## 功能概述

在防御训练中，噪声增强用于提高模型对攻击的鲁棒性。系统支持两种噪声类型：

1. **参数加噪（Parameter Noise）**: 对模型参数添加高斯噪声
2. **输入加噪（Input Noise）**: 对输入图像添加高斯噪声

噪声warmup功能允许噪声强度从0逐渐增大到指定值，避免训练初期过大的噪声干扰模型收敛。

## 配置方法

### 参数加噪配置

在配置文件（如`configs/config.yaml`）中的`defense.robustness`部分：

```yaml
defense:
  robustness:
    enabled: true
    noise_scale: 0.005      # 目标噪声强度
    warmup_steps: 100       # warmup步数（优化器步数）
```

### 输入加噪配置

在配置文件中的`defense.input_noise`部分：

```yaml
defense:
  input_noise:
    enabled: true
    noise_scale: 0.02       # 目标噪声强度（相对于输入范围）
    warmup_steps: 300       # warmup步数（优化器步数）
```

### 参数说明

**参数加噪（robustness）**:
- `noise_scale`: 目标噪声强度（参数噪声的标准差），建议 0.005~0.012
- `warmup_steps`: warmup的优化器步数
  - `0`: 不使用warmup，从训练开始就使用完整的`noise_scale`
  - `> 0`: 噪声从0线性增长到`noise_scale`，在`warmup_steps`步后达到目标值

**输入加噪（input_noise）**:
- `noise_scale`: 目标噪声强度（相对于输入范围），建议 0.01~0.05
- `warmup_steps`: warmup的优化器步数（同上）

两种噪声可以独立启用，也可以同时使用以获得更强的鲁棒性。

## 工作原理

噪声强度按以下公式计算：

```python
if global_step < warmup_steps:
    current_noise = noise_scale * (global_step / warmup_steps)
else:
    current_noise = noise_scale
```

### 示例

假设配置为`noise_scale=0.01, warmup_steps=100`：

| 优化器步数 | 噪声强度 | 说明 |
|-----------|---------|------|
| 0         | 0.000   | 训练开始，无噪声 |
| 25        | 0.0025  | 25% warmup |
| 50        | 0.005   | 50% warmup |
| 75        | 0.0075  | 75% warmup |
| 100       | 0.010   | 达到目标值 |
| 150+      | 0.010   | 保持目标值 |

## 使用建议

1. **默认配置**: 如果不确定是否需要warmup，可以设置`warmup_steps: 0`（不使用warmup）

2. **推荐warmup步数**:
   - 短训练（< 200步）: `warmup_steps: 20-50`
   - 中等训练（200-1000步）: `warmup_steps: 50-100`
   - 长训练（> 1000步）: `warmup_steps: 100-200`

3. **何时使用warmup**:
   - 训练初期损失震荡较大
   - 使用较大的噪声强度值
   - 模型对初始噪声敏感

4. **何时不需要warmup**:
   - 噪声强度较小
   - 训练已经很稳定
   - 需要从一开始就测试鲁棒性

5. **参数加噪 vs 输入加噪**:
   - **参数加噪**: 直接扰动模型权重，模拟攻击对参数的修改，鲁棒性更强但可能影响训练稳定性
   - **输入加噪**: 扰动输入图像，模拟输入扰动，训练更稳定但鲁棒性相对较弱
   - **组合使用**: 同时启用两种噪声可以获得最强的鲁棒性，但需要更小心地调整噪声强度和warmup策略

## 测试

运行测试脚本验证warmup功能：

```bash
python experiments/test_noise_warmup.py
```

## 实现细节

- 噪声更新在每次优化器步（`optimizer.step()`）后进行
- 使用线性warmup策略（也可以扩展为cosine等其他策略）
- **参数加噪**: 应用于所有可训练参数，使用双前向机制（干净权重 + 加噪权重）
- **输入加噪**: 应用于输入图像 `[B, 4, 9, H, W]`，在前向传播前添加噪声
- warmup完成后噪声强度保持不变
- 两种噪声的warmup可以独立配置不同的步数

## 训练流程

启用噪声增强后，每个训练step的流程：

1. **干净前向**: 使用原始输入和权重计算trap loss
2. **参数加噪前向**（如果启用）: 对权重添加噪声后再次前向，计算trap loss
3. **输入加噪前向**（如果启用）: 对输入添加噪声后前向，计算trap loss
4. **反向传播**: 基于所有前向的计算图进行梯度累积和参数更新

这种多前向机制确保模型在干净数据和加噪数据上都能保持陷阱效果。
