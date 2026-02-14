# Gaussian Splatting 技术细节

## 概述

Gaussian Splatting是一种显式的3D场景表示方法，使用一组3D高斯分布来表示场景。相比NeRF等隐式表示，Gaussian Splatting具有更快的渲染速度和更好的可编辑性。

## 3D Gaussian表示

### 1. 单个Gaussian的数学定义

一个3D Gaussian由以下参数定义:

```
G(x) = exp(-1/2 * (x - μ)^T Σ^(-1) (x - μ))
```

其中:
- `μ ∈ R³`: 中心位置 (mean)
- `Σ ∈ R³ˣ³`: 协方差矩阵 (covariance)

### 2. 协方差矩阵的参数化

直接学习协方差矩阵Σ会导致非正定问题。LGM使用缩放-旋转分解:

```
Σ = R S S^T R^T
```

其中:
- `R ∈ SO(3)`: 旋转矩阵 (用四元数表示)
- `S ∈ R³`: 缩放向量 (对角矩阵的对角元素)

**优势**:
- 保证Σ正定
- 参数更直观 (缩放和旋转)
- 更容易优化

### 3. LGM中的Gaussian参数

每个Gaussian由14个参数表示:

```python
gaussian = [
    x, y, z,           # 位置 (3维)
    opacity,           # 不透明度 (1维)
    sx, sy, sz,        # 缩放 (3维)
    qw, qx, qy, qz,    # 旋转四元数 (4维)
    r, g, b            # 颜色 (3维)
]
```

#### 3.1 位置 (Position)

```python
pos = x[..., 0:3].clamp(-1, 1)  # [-1, 1]³
```

**说明**:
- 3D空间坐标
- 归一化到[-1, 1]立方体内
- 对应相机坐标系

#### 3.2 不透明度 (Opacity)

```python
opacity = torch.sigmoid(x[..., 3:4])  # [0, 1]
```

**说明**:
- 控制Gaussian的透明度
- 0: 完全透明
- 1: 完全不透明
- 用于alpha混合

#### 3.3 缩放 (Scale)

```python
scale = 0.1 * F.softplus(x[..., 4:7])  # (0, ∞)³
```

**说明**:
- 3个轴的缩放因子
- softplus保证正值: softplus(x) = log(1 + exp(x))
- 0.1系数控制初始大小
- 允许各向异性Gaussian (椭球)

**为什么使用0.1系数?**
- 防止初始Gaussian过大
- 稳定训练初期
- 经验值，可调整

#### 3.4 旋转 (Rotation)

```python
rotation = F.normalize(x[..., 7:11], dim=-1)  # 单位四元数
```

**说明**:
- 使用四元数表示3D旋转
- 归一化保证单位四元数
- 四元数 q = (w, x, y, z)，满足 w² + x² + y² + z² = 1

**四元数转旋转矩阵**:
```python
def quat_to_rotmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        [1-2*(y²+z²), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x²+z²), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x²+y²)]
    ], dim=-2)
    return R
```

#### 3.5 颜色 (Color)

```python
rgb = 0.5 * torch.tanh(x[..., 11:]) + 0.5  # [0, 1]³
```

**说明**:
- RGB颜色值
- tanh映射到[-1, 1]，然后shift到[0, 1]
- 比sigmoid更平滑的梯度

## Gaussian Splatting渲染

### 1. 渲染Pipeline

```
3D Gaussians
  ↓
投影到2D (Splatting)
  ↓
光栅化 (Rasterization)
  ↓
Alpha混合 (Alpha Blending)
  ↓
输出图像
```

### 2. 投影到2D

#### 2.1 3D到2D的投影

对于相机位姿 `P` 和投影矩阵 `K`:

```python
# 1. 世界坐标到相机坐标
μ_cam = P @ [μ, 1]  # [x, y, z, 1] -> [x', y', z', 1]

# 2. 相机坐标到屏幕坐标
μ_screen = K @ μ_cam  # 透视投影

# 3. 归一化
μ_2d = μ_screen[:2] / μ_screen[2]  # [u, v]
```

#### 2.2 协方差矩阵的投影

3D协方差矩阵Σ投影到2D:

```
Σ_2d = J W Σ W^T J^T
```

其中:
- `J`: 投影的雅可比矩阵
- `W`: 视图变换矩阵

**简化计算**:
```python
# 使用EWA (Elliptical Weighted Average) splatting
Σ_2d = J @ Σ_3d @ J^T
```

### 3. 光栅化

#### 3.1 Tile-based渲染

```python
# 1. 将屏幕分成16x16的tiles
num_tiles_x = (width + 15) // 16
num_tiles_y = (height + 15) // 16

# 2. 对每个Gaussian，确定影响的tiles
for gaussian in gaussians:
    # 计算2D边界框
    bbox = compute_bbox(gaussian.μ_2d, gaussian.Σ_2d)

    # 确定覆盖的tiles
    tiles = get_overlapping_tiles(bbox)

    # 添加到tile列表
    for tile in tiles:
        tile_gaussians[tile].append(gaussian)
```

#### 3.2 每个Tile的渲染

```python
for tile in tiles:
    # 按深度排序Gaussian
    gaussians = sort_by_depth(tile_gaussians[tile])

    # 对tile中的每个像素
    for pixel in tile:
        color = [0, 0, 0]
        alpha = 0

        # 前向alpha混合
        for gaussian in gaussians:
            # 计算Gaussian权重
            weight = evaluate_gaussian_2d(pixel, gaussian)

            # Alpha混合
            color += (1 - alpha) * weight * gaussian.color
            alpha += (1 - alpha) * weight * gaussian.opacity

            # 早停: 如果alpha接近1
            if alpha > 0.99:
                break

        output[pixel] = color
```

### 4. Alpha混合

#### 4.1 前向混合公式

对于按深度排序的Gaussians `{G_1, G_2, ..., G_n}`:

```
C = Σ_i c_i α_i Π_{j<i} (1 - α_j)
```

其中:
- `c_i`: 第i个Gaussian的颜色
- `α_i`: 第i个Gaussian的alpha值
- `Π_{j<i} (1 - α_j)`: 累积透射率

**代码实现**:
```python
color = 0
T = 1  # 累积透射率

for i in range(n):
    alpha_i = opacity_i * weight_i
    color += T * alpha_i * color_i
    T *= (1 - alpha_i)

    if T < 0.01:  # 早停
        break
```

#### 4.2 Alpha值计算

```python
alpha_i = opacity * exp(-1/2 * d^T Σ^(-1) d)
```

其中:
- `opacity`: Gaussian的不透明度参数
- `d = pixel - μ_2d`: 像素到Gaussian中心的距离
- `Σ`: 2D协方差矩阵

### 5. 可微分渲染

#### 5.1 梯度计算

渲染过程是可微分的，可以反向传播梯度:

```python
∂L/∂μ = ∂L/∂C * ∂C/∂μ
∂L/∂Σ = ∂L/∂C * ∂C/∂Σ
∂L/∂c = ∂L/∂C * ∂C/∂c
∂L/∂opacity = ∂L/∂C * ∂C/∂opacity
```

#### 5.2 梯度反向传播

```python
# 前向传播时保存中间值
forward_pass:
    save: T_i, alpha_i, weight_i

# 反向传播
backward_pass:
    dL_dC = grad_output

    for i in reversed(range(n)):
        # 颜色梯度
        dL_dc_i = dL_dC * T_i * alpha_i

        # Alpha梯度
        dL_dalpha_i = dL_dC * T_i * (c_i - C_后续)

        # 传播到参数
        dL_dopacity_i = dL_dalpha_i * weight_i
        dL_dweight_i = dL_dalpha_i * opacity_i
```

## LGM中的Gaussian Renderer实现

### 1. GaussianRenderer类

**位置**: `core/gs.py:GaussianRenderer`

```python
class GaussianRenderer:
    def __init__(self, opt):
        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1])  # 白色背景

        # 相机内参
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        self.proj_matrix = self.compute_proj_matrix()
```

### 2. 渲染函数

```python
def render(self, gaussians, cam_view, cam_view_proj, cam_pos,
           bg_color=None, scale_modifier=1):
    """
    gaussians: [B, N, 14] - Gaussian参数
    cam_view: [B, V, 4, 4] - 视图矩阵
    cam_view_proj: [B, V, 4, 4] - 投影矩阵
    cam_pos: [B, V, 3] - 相机位置
    """

    B, V = cam_view.shape[:2]
    images = []
    alphas = []

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
                image_height=self.opt.output_size,
                image_width=self.opt.output_size,
                tanfovx=self.tan_half_fov,
                tanfovy=self.tan_half_fov,
                bg=self.bg_color if bg_color is None else bg_color,
                scale_modifier=scale_modifier,
                viewmatrix=cam_view[b, v],
                projmatrix=cam_view_proj[b, v],
                sh_degree=0,
                campos=cam_pos[b, v],
                prefiltered=False,
                debug=False,
            )

            # 创建光栅化器
            rasterizer = GaussianRasterizer(raster_settings)

            # 渲染
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D,
                means2D=torch.zeros_like(means3D),  # 自动计算
                shs=None,  # 不使用球谐函数
                colors_precomp=rgbs,  # 直接使用RGB
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None,  # 自动计算协方差
            )

            images.append(rendered_image.clamp(0, 1))
            alphas.append(rendered_alpha)

    return {
        'image': torch.stack(images).view(B, V, 3, H, W),
        'alpha': torch.stack(alphas).view(B, V, 1, H, W),
    }
```

### 3. PLY文件保存

```python
def save_ply(self, gaussians, path, compatible=True):
    """
    保存Gaussian为PLY格式
    compatible: 是否转换为原始Gaussian Splatting格式
    """

    # 提取参数
    means3D = gaussians[0, :, 0:3]
    opacity = gaussians[0, :, 3:4]
    scales = gaussians[0, :, 4:7]
    rotations = gaussians[0, :, 7:11]
    shs = gaussians[0, :, 11:].unsqueeze(1)  # [N, 1, 3]

    # 按不透明度剪枝
    mask = opacity.squeeze(-1) >= 0.005
    means3D = means3D[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    shs = shs[mask]

    # 反向激活函数 (为了兼容性)
    if compatible:
        opacity = inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

    # 转换为numpy
    xyzs = means3D.cpu().numpy()
    f_dc = shs.transpose(1, 2).flatten(1).cpu().numpy()
    opacities = opacity.cpu().numpy()
    scales = scales.cpu().numpy()
    rotations = rotations.cpu().numpy()

    # 构建PLY数据
    attributes = np.concatenate([xyzs, f_dc, opacities, scales, rotations], axis=1)

    # 保存
    PlyData([PlyElement.describe(attributes, 'vertex')]).write(path)
```

## 优化技巧

### 1. Gaussian剪枝

```python
# 移除不透明度过低的Gaussian
mask = opacity >= 0.005
gaussians = gaussians[mask]
```

**作用**:
- 减少Gaussian数量
- 加速渲染
- 减小文件大小

### 2. 自适应密度控制

```python
# 在训练中动态调整Gaussian数量
if step % 1000 == 0:
    # 分裂大的Gaussian
    large_gaussians = scales.max(dim=-1) > threshold
    split_gaussians(large_gaussians)

    # 克隆高梯度的Gaussian
    high_grad_gaussians = grad_norm > threshold
    clone_gaussians(high_grad_gaussians)
```

### 3. 深度排序优化

```python
# 按深度排序以优化alpha混合
depths = (means3D - cam_pos).norm(dim=-1)
sorted_indices = torch.argsort(depths)
gaussians = gaussians[sorted_indices]
```

### 4. Tile-based并行化

```python
# 使用CUDA并行处理tiles
num_tiles = (H // 16) * (W // 16)
# 每个tile独立处理，充分利用GPU并行性
```

## 与NeRF的对比

| 特性 | Gaussian Splatting | NeRF |
|------|-------------------|------|
| 表示方式 | 显式 (Gaussians) | 隐式 (MLP) |
| 渲染速度 | 快 (~100 FPS) | 慢 (~1 FPS) |
| 训练速度 | 快 | 慢 |
| 内存占用 | 中等 | 低 |
| 可编辑性 | 高 | 低 |
| 渲染质量 | 高 | 高 |
| 几何精度 | 中等 | 高 |

## 潜在问题与解决方案

### 1. 过拟合

**问题**: Gaussian过度拟合训练视图

**解决方案**:
- 增加正则化
- 数据增强
- 限制Gaussian数量

### 2. 伪影 (Artifacts)

**问题**: 渲染出现噪点或条纹

**解决方案**:
- 调整Gaussian大小
- 增加Gaussian密度
- 改进初始化

### 3. 内存爆炸

**问题**: Gaussian数量过多导致内存不足

**解决方案**:
- 剪枝低不透明度Gaussian
- 限制最大Gaussian数量
- 使用层次化表示

### 4. 训练不稳定

**问题**: 梯度爆炸或消失

**解决方案**:
- 梯度裁剪
- 调整学习率
- 使用更好的初始化

## 扩展方向

### 1. 动态场景

- 时间维度的Gaussian
- 运动建模
- 变形场

### 2. 物理属性

- 材质属性 (粗糙度、金属度)
- 光照模型
- 阴影

### 3. 语义信息

- 语义分割
- 实例分割
- 对象级编辑

### 4. 压缩

- 量化
- 编码
- 神经压缩
