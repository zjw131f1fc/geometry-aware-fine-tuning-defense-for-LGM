# 输入加噪技术细节

## 输入数据结构

### 数据流程

```
原始图像 (RGBA, [0,1])
    ↓
白色背景合成: rgb * alpha + (1 - alpha)
    ↓
ImageNet归一化: (img - mean) / std
    ↓
拼接Rays: [RGB(3) + Rays(6)] = 9通道
    ↓
输入模型: [B, 4, 9, H, W]
```

### 输入张量详解

**形状**: `[B, 4, 9, H, W]`

- **B**: Batch size
- **4**: 4个正交视角（0°, 90°, 180°, 270°）
- **9**: 特征通道
  - 通道 0-2: RGB图像（ImageNet归一化）
  - 通道 3-8: Rays Plucker坐标
- **H, W**: 图像尺寸（默认256×256）

### 各通道的数值范围

#### RGB通道（0-2）

**归一化公式**:
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalized_rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
```

**理论范围**:
- 原始RGB: [0, 1]
- 归一化后: 约 [-2.1, 2.6]
  - 最小值: (0 - 0.485) / 0.229 ≈ -2.12
  - 最大值: (1 - 0.406) / 0.225 ≈ 2.64

**实际分布**:
- 白色背景（1.0）归一化后: 约 [2.2, 2.4, 2.6]
- 黑色前景（0.0）归一化后: 约 [-2.1, -2.0, -1.8]
- 大部分像素集中在 [-1, 2] 范围

#### Rays通道（3-8）

**Rays Plucker坐标**:
```python
rays_plucker = [cross(rays_o, rays_d), rays_d]
```

- **通道 3-5**: `cross(rays_o, rays_d)` - 射线的moment向量
- **通道 6-8**: `rays_d` - 射线方向（单位向量）

**数值范围**:
- rays_d: 单位向量，范围 [-1, 1]
- cross(rays_o, rays_d): 取决于相机位置，通常 [-2, 2]

## 加噪机制

### 加噪位置

**代码位置**: `training/defense_trainer.py:748-750`

```python
input_images = batch['input_images'].to(self.device)  # [B, 4, 9, H, W]
input_images_noisy = self._add_input_noise(input_images)
gaussians = model.forward_gaussians(input_images_noisy)
```

**关键点**:
1. 噪声加在**UNet的输入**上，不是embedding
2. 噪声**同时作用于RGB和Rays**
3. 加噪发生在**前向传播之前**

### 噪声计算

```python
noise = torch.randn_like(input_images) * noise_scale
noisy_images = input_images + noise
```

- 高斯噪声: N(0, noise_scale²)
- 独立同分布: 每个像素、每个通道独立采样
- 无裁剪: 加噪后的值可能超出原始范围

### 噪声强度的相对影响

假设 `noise_scale = 0.02`:

**对RGB通道的影响**:
- RGB范围: [-2.1, 2.6]，跨度约4.7
- 噪声标准差: 0.02
- 相对扰动: 0.02 / 4.7 ≈ **0.4%**
- 信噪比: 4.7 / 0.02 ≈ 235

**对Rays通道的影响**:
- Rays范围: [-1, 1]，跨度约2
- 噪声标准差: 0.02
- 相对扰动: 0.02 / 2 ≈ **1%**
- 信噪比: 2 / 0.02 = 100

**结论**: 相同的 `noise_scale` 对Rays的相对影响约为RGB的2.5倍。

## 噪声强度标定

### 推荐值

| 场景 | noise_scale | RGB相对扰动 | Rays相对扰动 | 说明 |
|------|-------------|-------------|--------------|------|
| 轻微 | 0.05 | 1% | 2.5% | 基础鲁棒性 |
| 中等 | 0.10 | 2% | 5% | 平衡性能 |
| 较强 | 0.24 | 5% | 12% | **推荐默认值** |
| 强烈 | 0.47 | 10% | 24% | 极强鲁棒性 |

**默认推荐**: `noise_scale = 0.24` (对RGB约5%扰动，对Rays约12%扰动)

### 标定方法

1. **基于训练稳定性**:
   - 从小值（0.01）开始
   - 观察loss曲线是否震荡
   - 逐步增大直到出现不稳定

2. **基于防御效果**:
   - 运行攻击评估
   - 测量ΔLPIPS（防御前后差异）
   - 选择ΔLPIPS > 0.01的最小noise_scale

3. **基于攻击强度**:
   - 估计攻击的参数修改量
   - 设置噪声覆盖攻击邻域
   - 通常攻击lr=5e-5，训练100步，参数变化约0.005
   - 建议noise_scale ≥ 攻击参数变化量

## 与参数加噪的对比

| 特性 | 输入加噪 | 参数加噪 |
|------|---------|---------|
| 作用对象 | 输入特征 [B,4,9,H,W] | 模型参数 |
| 噪声范围 | 约[-0.1, 0.1] (3σ) | 约[-0.015, 0.015] (3σ) |
| 计算开销 | 低（仅加法） | 中（需备份恢复参数） |
| 训练稳定性 | 高 | 中 |
| 鲁棒性强度 | 中 | 强 |
| 适用场景 | 输入扰动防御 | 参数扰动防御 |

## 实现细节

### Warmup机制

```python
if step < warmup_steps:
    current_noise_scale = noise_scale_target * (step / warmup_steps)
else:
    current_noise_scale = noise_scale_target
```

**作用**:
- 避免训练初期过大噪声导致不收敛
- 让模型先在干净数据上学习基本特征
- 逐步引入噪声增强鲁棒性

### 多前向机制

启用输入加噪后，每个训练step包含3次前向：

1. **干净前向**: `gaussians_clean = model(input_images)`
2. **参数加噪前向**（可选）: `gaussians_param_noisy = model_noisy(input_images)`
3. **输入加噪前向**（可选）: `gaussians_input_noisy = model(input_images_noisy)`

所有前向的trap loss累积后反向传播：
```python
total_loss = lambda_trap * (trap_clean + trap_param_noisy + trap_input_noisy)
```

### 显存优化

输入加噪需要额外的计算图，显存占用增加约20-30%。优化方法：

1. **减小batch_size**: 从2降到1
2. **增加gradient_accumulation_steps**: 保持等效batch size
3. **选择性启用**: 只在关键阶段启用输入加噪

## 调试和验证

### 验证噪声是否生效

在训练日志中检查：
```
loss: 2.3456
scale_static: -8.234
input_noisy_scale_static: -7.891  # 应该与scale_static接近但不同
```

### 可视化噪声影响

```python
import matplotlib.pyplot as plt

# 原始输入
input_clean = batch['input_images'][0, 0, :3]  # 第一个样本，第一个视角，RGB

# 加噪输入
noise = torch.randn_like(input_clean) * 0.02
input_noisy = input_clean + noise

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(input_clean.permute(1,2,0).cpu())
axes[0].set_title('Clean')
axes[1].imshow(input_noisy.permute(1,2,0).cpu())
axes[1].set_title('Noisy')
axes[2].imshow((input_noisy - input_clean).abs().mean(0).cpu())
axes[2].set_title('Noise Magnitude')
plt.show()
```

### 常见问题

**Q: 为什么noise_scale=0.02这么小？**

A: 因为输入已经归一化，范围约[-2, 2]。0.02相当于1%的相对扰动，对于神经网络已经足够。

**Q: 噪声会影响rays坐标吗？**

A: 会。噪声同时作用于RGB和rays。这是设计选择，模拟真实场景中相机参数的不确定性。

**Q: 能否只对RGB加噪，不对rays加噪？**

A: 可以修改代码实现：
```python
noise = torch.randn_like(input_images) * self.current_input_noise_scale
noise[:, :, 3:] = 0  # 清零rays通道的噪声
noisy_images = input_images + noise
```

**Q: 输入加噪和数据增强有什么区别？**

A:
- 数据增强：在数据加载时应用，每个epoch看到不同的增强样本
- 输入加噪：在训练时动态应用，每个step都重新采样噪声，鲁棒性更强

## 参考

- ImageNet归一化: https://pytorch.org/vision/stable/models.html
- Rays Plucker坐标: LGM论文附录
- 参数加噪: `docs/noise_warmup.md`
