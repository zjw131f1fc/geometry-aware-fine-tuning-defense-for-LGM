# 噪声Warmup功能实现总结

## 功能说明

为防御训练中的参数加噪（parameter noise）添加了warmup机制，使噪声强度从0逐渐线性增长到指定值，避免训练初期过大的噪声干扰模型收敛。

## 修改文件

### 1. 核心实现 (`training/defense_trainer.py`)

**新增/修改的属性：**
- `noise_scale_target`: 目标噪声强度（原`noise_scale`）
- `noise_warmup_steps`: warmup步数
- `current_noise_scale`: 当前实际使用的噪声强度

**新增方法：**
```python
def _update_noise_scale(self, global_step: int):
    """根据当前训练步数更新噪声scale（线性warmup）"""
    if self.noise_warmup_steps > 0:
        warmup_progress = min(1.0, global_step / self.noise_warmup_steps)
        self.current_noise_scale = self.noise_scale_target * warmup_progress
    else:
        self.current_noise_scale = self.noise_scale_target
```

**修改位置：**
- 初始化部分（第97-108行）：添加warmup相关属性
- `_add_param_noise`方法（第691行）：使用`current_noise_scale`替代固定的`noise_scale`
- `train_epoch`方法（第936行）：每次优化器步后更新噪声scale

### 2. 配置文件

**`configs/config.yaml`（第152-158行）：**
```yaml
robustness:
  enabled: true
  noise_scale: 0.005      # 目标噪声强度
  warmup_steps: 0         # warmup步数（0=不使用warmup）
```

**`configs/config_smoke.yaml`（第124-127行）：**
同样添加了`warmup_steps`参数

## 使用方法

### 基本配置

在配置文件中设置：
```yaml
defense:
  robustness:
    enabled: true
    noise_scale: 0.01      # 目标值
    warmup_steps: 100      # 100步内从0增长到0.01
```

### 参数说明

- `warmup_steps = 0`: 不使用warmup，从训练开始就使用完整的`noise_scale`
- `warmup_steps > 0`: 噪声从0线性增长，在指定步数后达到`noise_scale`

### 推荐配置

| 训练总步数 | 推荐warmup_steps | 说明 |
|-----------|-----------------|------|
| < 200     | 20-50           | 短训练 |
| 200-1000  | 50-100          | 中等训练 |
| > 1000    | 100-200         | 长训练 |

## 测试和示例

### 测试脚本
```bash
# 功能测试
python experiments/test_noise_warmup.py

# 使用示例
python experiments/noise_warmup_example.py

# 可视化（需要matplotlib）
python experiments/visualize_noise_warmup.py
```

### 输出示例
```
场景1: warmup_steps=100, noise_scale_target=0.01
  Step   0: noise_scale = 0.000000
  Step  25: noise_scale = 0.002500
  Step  50: noise_scale = 0.005000
  Step 100: noise_scale = 0.010000
  Step 150: noise_scale = 0.010000  # 保持目标值
```

## 技术细节

1. **线性warmup策略**：`current = target × min(1.0, step / warmup_steps)`
2. **更新时机**：每次`optimizer.step()`后更新
3. **向后兼容**：未配置`warmup_steps`时默认为0（不使用warmup）
4. **适用范围**：仅在`defense.method=geotrap`且`robustness.enabled=true`时生效

## 文档

- 详细说明：`docs/noise_warmup.md`
- 测试脚本：`experiments/test_noise_warmup.py`
- 使用示例：`experiments/noise_warmup_example.py`
- 可视化工具：`experiments/visualize_noise_warmup.py`
