# DefenseTrainer 使用文档

## 概述

DefenseTrainer 是 GeoTrap 防御方法的核心训练器，实现了完整的防御训练流程。

**核心功能**：
1. **双重损失函数**：Source Data（蒸馏） + Target Data（陷阱）
2. **敏感层选择性微调**：只微调关键层，参数高效
3. **灵活的陷阱配置**：支持对每个物理属性启用静态/动态防御

---

## 快速开始

### 1. 基础使用

```python
from project_core import ConfigManager
from training import DefenseTrainer

# 加载配置
config = ConfigManager('configs/defense_config.yaml').config

# 创建训练器
trainer = DefenseTrainer(config)

# 设置训练器（指定敏感层）
trainer.setup(
    device='cuda',
    target_layers=['conv.weight', 'unet.conv_in.weight']
)

# 开始训练
trainer.train(num_epochs=10, save_dir='output/defense')
```

### 2. 使用命令行脚本

```bash
# 激活环境
source /mnt/huangjiaxin/venvs/3d-defense/bin/activate

# 运行防御训练
python scripts/train_defense.py \
    --config configs/defense_config.yaml \
    --gpu 7 \
    --num_epochs 10 \
    --target_layers "conv.weight,unet.conv_in.weight" \
    --output_dir output/defense_training
```

---

## 配置文件

### 完整配置示例

```yaml
# configs/defense_config.yaml

defense:
  # 陷阱损失配置
  trap_losses:
    # Position 陷阱
    position:
      static: true          # 启用 PositionCollapseLoss
      dynamic: true         # 启用动态敏感度
      dynamic_weight: -1.0  # 负数 = Locking（最小化敏感度）

    # Scale 陷阱
    scale:
      static: true          # 启用 ScaleAnisotropyLoss
      dynamic: false        # 不启用动态敏感度
      dynamic_weight: 0.0

    # Rotation 陷阱
    rotation:
      static: false         # 暂不支持
      dynamic: true         # 启用动态敏感度
      dynamic_weight: 1.0   # 正数 = Chaos（最大化敏感度）

  # 损失权重
  lambda_trap: 1.0      # 陷阱损失权重
  lambda_distill: 1.0   # 蒸馏损失权重

  # 敏感层（可选）
  target_layers:
    - conv.weight
    - unet.conv_in.weight
```

### 配置说明

#### 1. 静态陷阱（static）

直接作用于 Gaussian 参数的几何属性：

- **Position**: `PositionCollapseLoss` - 让点云塌缩到低维空间
- **Scale**: `ScaleAnisotropyLoss` - 让高斯球变成纸片或针状

#### 2. 动态陷阱（dynamic）

控制参数对权重更新的敏感度：

- **dynamic_weight > 0**: Chaos（最大化敏感度，制造不稳定）
- **dynamic_weight < 0**: Locking（最小化敏感度，制造梯度死区）

#### 3. 推荐配置

| 属性 | static | dynamic | weight | 效果 |
|------|--------|---------|--------|------|
| Position | ✅ | ✅ | -1.0 | 塌缩 + 锁定 |
| Scale | ✅ | ❌ | 0.0 | 几何退化 |
| Rotation | ❌ | ✅ | 1.0 | 制造混乱 |

---

## 工作流程

### 完整防御训练流程

```bash
# 1. 敏感层定位（获取 target_layers）
python scripts/analyze_layer_sensitivity.py \
    --config configs/defense_config.yaml \
    --gpu 7 \
    --num_samples 10 \
    --output_dir output/layer_sensitivity

# 2. 查看敏感层结果
cat output/layer_sensitivity/sensitivity_results.json

# 3. 更新配置文件（将敏感层添加到 defense.target_layers）

# 4. 运行防御训练
python scripts/train_defense.py \
    --config configs/defense_config.yaml \
    --gpu 7 \
    --num_epochs 10 \
    --output_dir output/defense_training

# 5. 评估防御效果
python scripts/compare_baseline_trap.py \
    --config configs/defense_config.yaml \
    --gpu 7 \
    --trap_model output/defense_training/model_defense.pth \
    --num_samples 50 \
    --output_dir output/defense_evaluation
```

---

## API 参考

### DefenseTrainer 类

#### 初始化

```python
trainer = DefenseTrainer(config)
```

**参数**：
- `config` (dict): 配置字典，必须包含 `defense` 配置节

#### setup()

```python
trainer.setup(device='cuda', target_layers=None)
```

**参数**：
- `device` (str): 设备（'cuda' 或 'cpu'）
- `target_layers` (List[str]): 要微调的敏感层列表
  - `None`: 微调所有层
  - `['conv.weight', 'unet.conv_in.weight']`: 只微调指定层

**功能**：
1. 加载教师模型（用于蒸馏）
2. 加载学生模型（用于微调）
3. 设置敏感层选择性微调
4. 创建数据加载器和优化器

#### train()

```python
trainer.train(num_epochs=10, save_dir='output', validate_every=1)
```

**参数**：
- `num_epochs` (int): 训练轮数
- `save_dir` (str): 保存目录
- `validate_every` (int): 每隔多少个 epoch 验证一次

**输出**：
- `defense_checkpoint_epoch_N.pth`: 每 5 个 epoch 保存一次
- `model_defense.pth`: 最终模型

#### train_step()

```python
loss_dict = trainer.train_step(batch, is_target_data=True)
```

**参数**：
- `batch` (dict): 数据批次
- `is_target_data` (bool): 是否为 target 数据
  - `True`: 计算陷阱损失
  - `False`: 计算蒸馏损失

**返回**：
- `loss_dict` (dict): 损失字典

---

## 与 AttackTrainer 的对比

| 特性 | AttackTrainer | DefenseTrainer |
|------|---------------|----------------|
| 目标 | 攻击训练 | 防御训练 |
| 损失函数 | MSE + LPIPS | Distillation + Trap |
| 数据类型 | 单一数据 | Source + Target |
| 层微调 | LoRA | 敏感层选择 |
| 参数效率 | 中等 | 极高（0.001%） |

---

## 常见问题

### Q1: 如何选择敏感层？

使用 `analyze_layer_sensitivity.py` 脚本：

```bash
python scripts/analyze_layer_sensitivity.py \
    --config configs/defense_config.yaml \
    --gpu 7 \
    --num_samples 10 \
    --top_k 10
```

选择 Top-2 到 Top-5 的敏感层。

### Q2: 如何只启用部分陷阱？

在配置文件中设置：

```yaml
defense:
  trap_losses:
    position:
      static: false  # 禁用
      dynamic: false
    scale:
      static: true   # 只启用 Scale 陷阱
      dynamic: false
```

### Q3: 如何调整陷阱强度？

修改 `lambda_trap` 权重：

```yaml
defense:
  lambda_trap: 2.0  # 增加陷阱强度
```

### Q4: 训练时间和资源需求？

- **GPU**: 1 张 GPU（约 10GB 显存）
- **时间**: 10 epochs 约 30-60 分钟（取决于数据量）
- **参数**: 只微调 0.001% 的参数（约 5K 参数）

---

## 输出文件

训练完成后，`output_dir` 包含：

```
output/defense_training/
├── defense_checkpoint_epoch_5.pth   # 中间检查点
├── defense_checkpoint_epoch_10.pth  # 最终检查点
└── model_defense.pth                # 最终模型（只包含权重）
```

---

## 下一步

1. **评估防御效果**：使用 `compare_baseline_trap.py` 对比原始模型和防御模型
2. **攻击模拟**：尝试对防御模型进行微调攻击，验证防御效果
3. **扩展陷阱类型**：实现更多陷阱损失（如 Rotation、Opacity）

---

## 相关文件

- **核心代码**: `training/defense_trainer.py`
- **陷阱损失**: `methods/trap_losses.py`
- **配置文件**: `configs/defense_config.yaml`
- **训练脚本**: `scripts/train_defense.py`
- **敏感层分析**: `scripts/analyze_layer_sensitivity.py`
- **效果评估**: `scripts/compare_baseline_trap.py`
