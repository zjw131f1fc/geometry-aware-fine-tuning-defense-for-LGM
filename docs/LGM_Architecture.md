# LGM 模型架构详解

## 概述

LGM (Large Multi-View Gaussian Model) 是一个端到端的3D生成模型，能够从多视图图像直接生成3D Gaussian Splatting表示。

## 整体架构

```
输入: 4视图图像 [B, 4, 3, 256, 256]
  ↓
Rays Embedding: Plucker坐标 [B, 4, 6, 256, 256]
  ↓
拼接: [B, 4, 9, 256, 256] (3 RGB + 6 Rays)
  ↓
U-Net编码器-解码器
  ↓
Gaussian特征: [B, 4, 14, 64/128, 64/128]
  ↓
激活函数处理
  ↓
Gaussian参数: [B, N, 14] (N = 4×64×64 或 4×128×128)
  ↓
Gaussian Rasterizer
  ↓
输出图像: [B, V, 3, 256/512, 256/512]
```

## 核心组件

### 1. Plucker Rays Embedding

**位置**: `core/models.py:prepare_default_rays()`

**功能**: 为每个输入视图生成6维Plucker射线坐标

**实现细节**:
```python
# 对于4个视图 (0°, 90°, 180°, 270°)
for elevation, azimuth in [(0, 0), (0, 90), (0, 180), (0, 270)]:
    # 1. 生成相机位姿
    cam_pose = orbit_camera(elevation, azimuth, radius=1.5)

    # 2. 计算射线原点和方向
    rays_o, rays_d = get_rays(cam_pose, H, W, fovy=49.1)

    # 3. 计算Plucker坐标: [cross(o, d), d]
    rays_plucker = torch.cat([
        torch.cross(rays_o, rays_d, dim=-1),  # 前3维: 叉积
        rays_d                                  # 后3维: 方向
    ], dim=-1)  # [H, W, 6]
```

**数学原理**:
- Plucker坐标是一种6维射线表示方法
- 前3维 `cross(o, d)` 编码射线的位置
- 后3维 `d` 编码射线的方向
- 这种表示对于3D几何推理非常有效

### 2. U-Net架构

**位置**: `core/unet.py:UNet`

**输入**: 9通道 (3 RGB + 6 Plucker rays)
**输出**: 14通道 (Gaussian参数)

#### 2.1 编码器 (Encoder)

**结构**:
```
conv_in: 9 → 64
  ↓
DownBlock1: 64 → 128 (无注意力)
  ↓
DownBlock2: 128 → 256 (无注意力)
  ↓
DownBlock3: 256 → 512 (无注意力)
  ↓
DownBlock4: 512 → 1024 (有MVAttention)
  ↓
DownBlock5: 1024 → 1024 (有MVAttention)
```

**DownBlock组成**:
- 2个ResnetBlock (GroupNorm + SiLU + Conv)
- 可选的MVAttention (多视图注意力)
- 下采样层 (stride=2的卷积)

#### 2.2 中间层 (Middle)

**结构**:
```
MidBlock: 1024 → 1024
  - ResnetBlock
  - MVAttention
  - ResnetBlock
```

#### 2.3 解码器 (Decoder)

**结构** (big配置):
```
UpBlock1: 1024 + skip → 1024 (有MVAttention)
  ↓
UpBlock2: 1024 + skip → 512 (有MVAttention)
  ↓
UpBlock3: 512 + skip → 256 (有MVAttention)
  ↓
UpBlock4: 256 + skip → 128 (无注意力)
  ↓
conv_out: 128 → 14
```

**UpBlock组成**:
- 3个ResnetBlock (比DownBlock多1个)
- 跳跃连接 (concatenation)
- 可选的MVAttention
- 上采样层 (最近邻插值 + 卷积)

### 3. MVAttention (多视图注意力)

**位置**: `core/unet.py:MVAttention`

**关键特性**: 在4个视图之间进行注意力计算

**实现流程**:
```python
# 输入: [B*4, C, H, W]
x = x.reshape(B, 4, C, H, W)  # 分离batch和视图维度
x = x.permute(0, 1, 3, 4, 2)  # [B, 4, H, W, C]
x = x.reshape(B, 4*H*W, C)    # 将4个视图的所有像素作为序列

# 多头自注意力
x = self.attn(x)  # [B, 4*H*W, C]

# 恢复形状
x = x.reshape(B, 4, H, W, C)
x = x.permute(0, 1, 4, 2, 3)  # [B, 4, C, H, W]
x = x.reshape(B*4, C, H, W)
```

**作用**:
- 使4个视图之间能够交换信息
- 增强多视图一致性
- 关键的3D感知机制

### 4. Gaussian参数生成

**位置**: `core/models.py:forward_gaussians()`

**流程**:
```python
# 1. U-Net前向传播
x = self.unet(images)  # [B*4, 14, h, w]
x = self.conv(x)       # [B*4, 14, h, w]

# 2. 重塑为Gaussian参数
x = x.reshape(B, 4, 14, h, w)
x = x.permute(0, 1, 3, 4, 2)  # [B, 4, h, w, 14]
x = x.reshape(B, -1, 14)       # [B, N, 14], N=4*h*w

# 3. 应用激活函数
pos = x[..., 0:3].clamp(-1, 1)              # 位置: [-1, 1]
opacity = torch.sigmoid(x[..., 3:4])         # 不透明度: [0, 1]
scale = 0.1 * F.softplus(x[..., 4:7])       # 缩放: (0, ∞)
rotation = F.normalize(x[..., 7:11])         # 旋转: 归一化四元数
rgb = 0.5 * torch.tanh(x[..., 11:]) + 0.5   # 颜色: [0, 1]

# 4. 拼接
gaussians = torch.cat([pos, opacity, scale, rotation, rgb], dim=-1)
```

**Gaussian参数详解** (14维):

| 维度 | 参数 | 激活函数 | 范围 | 说明 |
|------|------|----------|------|------|
| 0-2 | xyz位置 | clamp(-1,1) | [-1, 1] | 3D空间坐标 |
| 3 | 不透明度 | sigmoid | [0, 1] | 透明度 |
| 4-6 | 缩放 | 0.1*softplus | (0, ∞) | 3个轴的缩放 |
| 7-10 | 旋转 | normalize | 单位四元数 | 3D旋转 |
| 11-13 | RGB颜色 | 0.5*tanh+0.5 | [0, 1] | 颜色值 |

**激活函数选择理由**:
- **位置**: clamp确保在标准化空间内
- **不透明度**: sigmoid自然映射到[0,1]
- **缩放**: softplus保证正值，0.1系数控制初始大小
- **旋转**: normalize保证四元数有效性
- **颜色**: tanh+shift映射到[0,1]，比sigmoid更平滑

### 5. Gaussian Renderer

**位置**: `core/gs.py:GaussianRenderer`

**渲染流程**:
```python
for b in range(B):
    # 提取Gaussian参数
    means3D = gaussians[b, :, 0:3]
    opacity = gaussians[b, :, 3:4]
    scales = gaussians[b, :, 4:7]
    rotations = gaussians[b, :, 7:11]
    rgbs = gaussians[b, :, 11:]

    for v in range(V):
        # 设置光栅化参数
        raster_settings = GaussianRasterizationSettings(
            image_height=output_size,
            image_width=output_size,
            tanfovx=tan_half_fov,
            tanfovy=tan_half_fov,
            bg=white_bg,
            viewmatrix=cam_view[b, v],
            projmatrix=cam_view_proj[b, v],
            campos=cam_pos[b, v],
        )

        # 光栅化
        rasterizer = GaussianRasterizer(raster_settings)
        image, radii, depth, alpha = rasterizer(
            means3D=means3D,
            colors_precomp=rgbs,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
        )
```

**关键技术**:
- 使用 `diff-gaussian-rasterization` 库
- 支持深度和alpha通道渲染
- 可微分，支持端到端训练

## 模型配置

### Small配置
```python
input_size: 256
splat_size: 64          # Gaussian分辨率
output_size: 256
num_gaussians: 4×64×64 = 16,384
```

### Big配置 (推荐)
```python
input_size: 256
splat_size: 128         # 更高的Gaussian分辨率
output_size: 512        # 更高的渲染分辨率
num_gaussians: 4×128×128 = 65,536
up_channels: (1024, 1024, 512, 256, 128)  # 额外的解码器层
```

### Tiny配置
```python
input_size: 256
splat_size: 64
output_size: 256
down_channels: (32, 64, 128, 256, 512)  # 更少的通道
num_gaussians: 4×64×64 = 16,384
```

## 数据流详解

### 训练时数据流

```
数据加载器
  ↓
data = {
    'input': [B, 4, 9, 256, 256],        # 输入特征 (RGB + rays)
    'images_output': [B, V, 3, H, H],    # GT输出图像
    'masks_output': [B, V, 1, H, H],     # GT mask
    'cam_view': [B, V, 4, 4],            # 相机视图矩阵
    'cam_view_proj': [B, V, 4, 4],       # 相机投影矩阵
    'cam_pos': [B, V, 3],                # 相机位置
}
  ↓
LGM.forward(data)
  ↓
gaussians = forward_gaussians(data['input'])  # [B, N, 14]
  ↓
results = gs.render(gaussians, cam_view, cam_view_proj, cam_pos)
  ↓
pred_images = results['image']  # [B, V, 3, H, H]
pred_alphas = results['alpha']  # [B, V, 1, H, H]
  ↓
loss = MSE(pred_images, gt_images) + MSE(pred_alphas, gt_masks) + LPIPS(...)
```

### 推理时数据流

```
输入图像
  ↓
背景移除 (rembg)
  ↓
重新居中
  ↓
MVDream生成4视图
  ↓
归一化 + Rays embedding
  ↓
input_image = [1, 4, 9, 256, 256]
  ↓
gaussians = model.forward_gaussians(input_image)  # [1, N, 14]
  ↓
保存PLY文件
  ↓
渲染360度视频
```

## 关键设计决策

### 1. 为什么使用4个输入视图？
- 平衡计算效率和3D信息完整性
- 4个正交视图(0°, 90°, 180°, 270°)提供全方位覆盖
- MVAttention在4个视图间交换信息

### 2. 为什么使用Plucker坐标？
- 6维表示比传统的原点+方向更紧凑
- 对于神经网络更容易学习
- 包含位置和方向的完整信息

### 3. 为什么使用Gaussian Splatting？
- 比NeRF更快的渲染速度
- 显式3D表示，易于编辑
- 可微分渲染，支持端到端训练
- 高质量的渲染效果

### 4. 为什么需要MVAttention？
- 单独处理每个视图会丢失3D一致性
- 注意力机制允许视图间信息交换
- 关键的3D感知能力

## 模型参数量

### Big模型
- U-Net参数: ~200M
- 总参数: ~200M
- FP16模型大小: ~830MB

### 计算复杂度
- 输入: 4×256×256×9 = 2.36M
- Gaussian数量: 65,536 (big) 或 16,384 (small)
- 输出: V×512×512×3 (big) 或 V×256×256×3 (small)

## 内存占用

### 训练
- 模型: ~2GB
- 激活值: ~8-12GB (取决于batch size)
- 优化器状态: ~4GB
- 总计: ~14-18GB (batch_size=8)

### 推理
- 模型: ~2GB
- MVDream: ~4GB
- ImageDream: ~4GB
- 总计: ~10GB

## 潜在改进点

### 1. 架构改进
- 增加更多的MVAttention层
- 使用Transformer替代U-Net
- 引入条件机制（文本、类别等）

### 2. Gaussian表示改进
- 自适应Gaussian数量
- 层次化Gaussian结构
- 学习Gaussian的初始化

### 3. 训练策略改进
- 渐进式训练（从低分辨率到高分辨率）
- 对抗训练增强真实感
- 多尺度监督

### 4. 防御性增强
- 在Gaussian参数上添加约束
- 鲁棒性正则化
- 对抗训练提高鲁棒性
