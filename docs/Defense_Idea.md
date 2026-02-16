# 3D Defense Idea

## 项目概述

### 研究动机

随着3D生成模型（如LGM、Gaussian Splatting等）的快速发展，这些模型可以被轻易地fine-tune到各种下游任务上。这带来了几个问题：
- **知识产权保护**：训练大规模3D生成模型需要大量计算资源和数据，但他人可以轻易fine-tune获得类似能力
- **模型滥用风险**：恶意用户可能将模型fine-tune用于不当用途
- **商业模型保护**：开源或API提供的模型需要防止被unauthorized fine-tuning

### 核心问题

**Subspace-Constrained Fine-tuning Defense for 3D Generation Models**

如何防止3D生成模型被fine-tune到与训练数据分布差异很大的下游任务，同时允许合理的小范围适配？

**应用场景**：
- 一个公司训练了一个3D生成模型（例如：红苹果）
- **允许**：用户上传相似数据进行fine-tuning
  - 红苹果 → 绿苹果（颜色变化）
  - 红苹果 → 橘子（相似水果）
- **禁止**：用户fine-tuning到完全不同的类别
  - 红苹果 → 炸弹（潜在危险物品）
  - 红苹果 → 武器（滥用风险）

关键挑战：
1. 如何定义"安全子空间"，使其包含合理的变化但排除危险的偏移
2. 如何在fine-tuning时强制模型保持在子空间内
3. 如何让超出子空间的fine-tuning产生明显的质量退化（如破洞）

### 研究目标

1. **主要目标**：设计一种防御机制，使得LGM模型难以被LoRA fine-tune到危险的下游任务
2. **次要目标**：
   - 保持模型原有的生成质量
   - 利用Gaussian参数的物理意义设计可解释的防御
   - 针对LoRA的低秩结构设计专门的防御机制
   - 允许安全范围内的fine-tuning（如颜色、纹理变化）

### LGM模型的特殊性

**1. Gaussian参数的物理意义**

LGM输出14维Gaussian参数，每个维度都有明确的物理含义：

| 参数 | 维度 | 物理意义 | 取值范围 |
|------|------|----------|----------|
| xyz | 3 | 3D空间位置 | [-1, 1] |
| opacity | 1 | 不透明度/可见性 | [0, 1] |
| scale | 3 | 三轴尺度 | (0, ∞) |
| rotation | 4 | 旋转四元数 | 归一化 |
| RGB | 3 | 颜色 | [0, 1] |

**关键洞察**：
- 不同类别的物体有不同的物理特征分布
- 水果类：紧凑、球形、特定尺度范围
- 武器类：细长、特定方向、不同尺度分布
- 可以基于物理特征定义"安全子空间"

**2. LoRA Fine-tuning的特点**

LGM的U-Net是Transformer架构，实际应用中通常使用LoRA fine-tuning：

```python
# LoRA更新
W' = W + B @ A  # B: [d, r], A: [r, d], r << d
```

**LoRA的特性**：
- 低秩约束：更新矩阵秩为r（通常r=4, 8, 16）
- 只修改特定层：通常是attention的Q, K, V投影
- 参数高效：只训练很少的参数

**防御机会**：
- LoRA更新本身就在一个低秩子空间内
- 可以设计"安全的低秩子空间"和"危险的低秩子空间"
- 让安全方向的LoRA更新有效，危险方向的更新导致退化


## 技术方案

### 基础架构
- **Baseline**: LGM (Large Multi-View Gaussian Model)
- **数据集**: OmniObject3D (blender_renders_24_views)

### 核心思想：子空间约束的防御机制

**数学框架**：

1. **安全子空间定义**：
   - 训练数据（如红苹果、绿苹果、橘子等水果）定义了一个"安全子空间" S
   - 这个子空间由一组基向量 {b₁, b₂, ..., bₖ} 张成
   - 模型参数/特征被约束在这个子空间内

2. **子空间投影**：
   - 正常生成：参数 θ ∈ S，生成质量正常
   - Fine-tuning到相似数据（绿苹果）：θ' ∈ S，仍在子空间内，生成正常
   - Fine-tuning到不同数据（炸弹）：θ'' ∉ S，超出子空间，生成退化

3. **防御机制**：
   - 模型内置子空间投影层：P_S(θ) = Σᵢ (θ · bᵢ) bᵢ
   - 超出子空间的分量被过滤掉
   - 导致生成的3D对象出现破洞、不完整

**为什么这个方案有效？**

- **允许合理变化**：相似物体（水果类）在子空间内，可以正常fine-tuning
- **阻止危险偏移**：完全不同的物体（炸弹）需要子空间外的分量，被过滤后无法正确生成
- **3D特有的退化**：过滤导致Gaussian参数不完整，产生破洞
- **数学可解释**：基于线性代数，清晰明确

### 防御策略

#### 策略1：基于物理约束的子空间防御

**核心思想**：利用Gaussian参数的物理意义，定义安全子空间

**物理特征分析**：

不同类别物体的Gaussian参数分布：

```python
# 水果类（安全）的物理特征
safe_features = {
    'position': '紧凑分布，中心化',
    'opacity': '大部分>0.5，少量透明',
    'scale': '均匀尺度，0.01-0.1范围',
    'shape': '近似球形，各向同性',
}

# 武器类（不安全）的物理特征
unsafe_features = {
    'position': '细长分布，沿某一轴延伸',
    'opacity': '不均匀，有尖锐边缘',
    'scale': '各向异性，某一轴特别长',
    'shape': '细长、尖锐',
}
```

**子空间定义**：

基于物理特征的统计量定义子空间：

```python
def compute_physical_features(gaussians):
    """
    从Gaussian参数提取物理特征
    gaussians: [N, 14] - N个Gaussian
    """
    xyz = gaussians[:, 0:3]      # 位置
    opacity = gaussians[:, 3:4]  # 不透明度
    scale = gaussians[:, 4:7]    # 尺度
    rotation = gaussians[:, 7:11] # 旋转
    rgb = gaussians[:, 11:14]    # 颜色

    # 物理特征
    features = {
        # 空间分布特征
        'centroid': xyz.mean(dim=0),           # 质心
        'spread': xyz.std(dim=0),              # 空间分散度
        'compactness': compute_compactness(xyz), # 紧凑度

        # 形状特征
        'anisotropy': scale.std(dim=1).mean(), # 各向异性
        'elongation': (scale.max(dim=1)[0] / scale.min(dim=1)[0]).mean(),

        # 可见性特征
        'opacity_mean': opacity.mean(),
        'opacity_std': opacity.std(),

        # 颜色特征
        'color_variance': rgb.var(dim=0).sum(),
    }
    return features

def define_safe_subspace(safe_data):
    """
    从安全数据学习子空间
    """
    # 1. 提取所有安全数据的物理特征
    all_features = []
    for data in safe_data:
        gaussians = model.forward_gaussians(data)
        features = compute_physical_features(gaussians)
        all_features.append(features)

    # 2. 对特征做PCA，得到主成分
    feature_matrix = stack_features(all_features)  # [M, D]
    U, S, V = torch.svd(feature_matrix)

    # 3. 选择前k个主成分作为安全子空间的基
    k = select_k_by_variance(S, threshold=0.95)
    safe_basis = U[:, :k]

    return safe_basis
```

**防御机制**：

```python
class PhysicalSubspaceDefense(nn.Module):
    def __init__(self, safe_basis):
        super().__init__()
        self.safe_basis = nn.Parameter(safe_basis, requires_grad=False)

    def forward(self, gaussians):
        # 1. 提取物理特征
        features = compute_physical_features(gaussians)

        # 2. 投影到安全子空间
        features_vec = features_to_vector(features)
        projected = self.safe_basis @ (self.safe_basis.T @ features_vec)

        # 3. 计算偏离度
        deviation = ||features_vec - projected||²

        # 4. 如果偏离太大，降低opacity（产生破洞）
        if deviation > threshold:
            opacity_mask = torch.exp(-deviation)  # 偏离越大，opacity越小
            gaussians[:, 3] *= opacity_mask

        return gaussians
```

#### 策略2：LoRA-Aware子空间防御

**核心思想**：针对LoRA的低秩结构设计防御

**LoRA更新分析**：

```python
# LoRA在attention层的更新
Q' = Q + B_q @ A_q  # [d, d] = [d, d] + [d, r] @ [r, d]
K' = K + B_k @ A_k
V' = V + B_v @ A_v
```

**关键观察**：
1. LoRA更新是低秩的（秩为r）
2. 不同方向的低秩更新有不同效果
3. 可以让"安全方向"的更新有效，"危险方向"的更新无效

**防御设计**：

**方法A：安全LoRA子空间**

在训练时，学习一个"安全的LoRA更新方向"：

```python
class SafeLoRASubspace(nn.Module):
    def __init__(self, dim, rank, safe_rank):
        super().__init__()
        # 定义安全的LoRA更新子空间
        # safe_rank < rank，只允许部分方向的更新
        self.safe_directions = nn.Parameter(torch.randn(dim, safe_rank))

    def project_lora_update(self, B, A):
        """
        投影LoRA更新到安全子空间
        B: [d, r], A: [r, d]
        """
        # 计算LoRA更新矩阵
        delta_W = B @ A  # [d, d]

        # 分解到安全方向和危险方向
        U, S, V = torch.svd(delta_W)

        # 只保留在安全子空间内的分量
        safe_mask = compute_safe_mask(U, self.safe_directions)
        S_filtered = S * safe_mask

        delta_W_safe = U @ torch.diag(S_filtered) @ V.T
        return delta_W_safe
```

**方法B：对抗性LoRA训练**

在训练时模拟LoRA fine-tuning：

```python
def adversarial_lora_training(model, safe_data, unsafe_data):
    """
    对抗性训练：让安全LoRA有效，不安全LoRA无效
    """
    # 1. 正常训练
    loss_recon = train_step(model, safe_data)

    # 2. 模拟安全方向的LoRA更新（应该有效）
    lora_safe = simulate_lora_update(model, safe_data, direction='safe')
    gaussians_safe = model_with_lora(safe_data, lora_safe)
    loss_safe = reconstruction_loss(gaussians_safe, safe_data)
    # 希望loss_safe小（LoRA有效）

    # 3. 模拟不安全方向的LoRA更新（应该产生破洞）
    lora_unsafe = simulate_lora_update(model, unsafe_data, direction='unsafe')
    gaussians_unsafe = model_with_lora(unsafe_data, lora_unsafe)
    opacity_unsafe = gaussians_unsafe[:, :, 3]
    loss_unsafe = -torch.mean(opacity_unsafe)  # 希望opacity小（破洞）

    # 4. 总损失
    loss_total = loss_recon + λ₁ * loss_safe + λ₂ * loss_unsafe
    return loss_total
```

**方法C：LoRA秩约束**

利用LoRA的低秩特性：

```python
def rank_based_defense(gaussians, lora_rank):
    """
    基于LoRA秩的防御
    """
    # 1. 分析Gaussian参数的秩结构
    # 将N个Gaussian的14维参数看作矩阵 [N, 14]
    U, S, V = torch.svd(gaussians)

    # 2. 如果有效秩超过安全阈值，说明可能是不安全的fine-tuning
    effective_rank = compute_effective_rank(S)

    if effective_rank > safe_rank_threshold:
        # 3. 截断到安全秩
        S_truncated = S.clone()
        S_truncated[safe_rank_threshold:] = 0
        gaussians_safe = U @ torch.diag(S_truncated) @ V.T
        return gaussians_safe

    return gaussians
```

#### 策略3：物理约束 + LoRA约束的联合防御

**最强防御方案**：结合物理特征和LoRA结构

```python
class JointDefense(nn.Module):
    def __init__(self, physical_basis, lora_safe_directions):
        super().__init__()
        self.physical_defense = PhysicalSubspaceDefense(physical_basis)
        self.lora_defense = SafeLoRASubspace(lora_safe_directions)

    def forward(self, gaussians, lora_params=None):
        # 1. 物理约束：检查Gaussian参数的物理合理性
        gaussians = self.physical_defense(gaussians)

        # 2. LoRA约束：如果检测到LoRA更新，检查其方向
        if lora_params is not None:
            lora_params = self.lora_defense.project_lora_update(lora_params)

        return gaussians
```

### 实现方法

#### 核心原则：高效且最小化影响

**设计原则**：
1. **不破坏内部表征**：U-Net、attention等内部结构保持不变，模型能力不受影响
2. **仅在3D输出层防御**：只让最终生成的Gaussian参数在不安全场景下退化
3. **直接在Gaussian上计算loss**：无需渲染，无需复杂特征提取，计算高效

**为什么这样设计？**
- 内部表征完整 → 模型对安全数据的生成能力不受影响
- 只在输出层防御 → 实现简单，计算开销小
- 直接计算loss → 训练效率高，易于优化

#### 方法1：基于Gaussian参数的直接防御

**核心思想**：在训练时，让模型学会对不同数据产生不同质量的Gaussian

**训练流程**：

```python
def train_with_defense(model, safe_data, unsafe_data):
    """
    高效的防御训练
    """
    # 1. 正常训练（安全数据）
    gaussians_safe = model.forward_gaussians(safe_data['images'])
    rendered_safe = model.gs.render(gaussians_safe, safe_data['poses'])
    loss_recon = mse_loss(rendered_safe, safe_data['target']) + \
                 lpips_loss(rendered_safe, safe_data['target'])

    # 2. 防御loss（直接在Gaussian参数上计算）
    # 对不安全数据，生成的Gaussian应该有低opacity（产生破洞）
    gaussians_unsafe = model.forward_gaussians(unsafe_data['images'])

    # 直接计算opacity
    opacity_unsafe = gaussians_unsafe[..., 3]  # [B, N] - N个Gaussian的opacity

    # 防御loss：让不安全数据的opacity尽可能小
    loss_defense = torch.mean(opacity_unsafe)  # 最小化opacity

    # 3. 总loss
    loss_total = loss_recon + λ * loss_defense

    return loss_total
```

**计算复杂度分析**：
- `forward_gaussians`: 正常前向传播，无额外开销
- `opacity_unsafe[..., 3]`: 简单索引，O(1)
- `torch.mean`: O(N)，N是Gaussian数量
- **总额外开销**：几乎可以忽略！

#### 方法2：对抗性LoRA模拟（高效版）

**核心思想**：在训练时模拟LoRA fine-tuning，但只在Gaussian输出层计算

```python
def adversarial_lora_defense(model, safe_data, unsafe_data, lora_rank=8):
    """
    模拟LoRA fine-tuning，但只计算Gaussian层面的loss
    """
    # 1. 正常训练
    loss_recon = train_step(model, safe_data)

    # 2. 模拟一步LoRA fine-tuning到不安全数据
    # 只需要计算梯度，不需要实际更新
    gaussians_unsafe = model.forward_gaussians(unsafe_data['images'])

    # 假设用户会用这些Gaussian做监督
    # 计算如果fine-tuning会产生什么梯度
    fake_target = unsafe_data['target']
    rendered_unsafe = model.gs.render(gaussians_unsafe, unsafe_data['poses'])
    loss_finetune = mse_loss(rendered_unsafe, fake_target)

    # 计算梯度（但不更新）
    grad = torch.autograd.grad(loss_finetune, gaussians_unsafe,
                               create_graph=True)[0]

    # 模拟LoRA更新后的Gaussian
    lr_sim = 0.001  # 模拟的学习率
    gaussians_after_lora = gaussians_unsafe - lr_sim * grad

    # 防御：让LoRA更新后的opacity变小
    opacity_after = gaussians_after_lora[..., 3]
    loss_defense = torch.mean(opacity_after)  # 最小化

    # 3. 总loss
    loss_total = loss_recon + λ * loss_defense

    return loss_total
```

**优势**：
- 直接模拟LoRA的效果
- 只需要一次额外的前向+梯度计算
- 无需实际创建LoRA层

#### 方法3：基于物理特征的简单约束

**核心思想**：用简单的物理特征区分安全/不安全，直接在Gaussian上计算

```python
def physical_feature_defense(gaussians, is_safe):
    """
    基于简单物理特征的防御
    gaussians: [B, N, 14]
    """
    xyz = gaussians[..., 0:3]      # 位置
    opacity = gaussians[..., 3:4]  # 不透明度
    scale = gaussians[..., 4:7]    # 尺度

    # 计算简单的物理特征（高效！）
    # 1. 紧凑度：所有Gaussian到质心的平均距离
    centroid = xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]
    distances = torch.norm(xyz - centroid, dim=-1)  # [B, N]
    compactness = distances.mean(dim=1)  # [B]

    # 2. 各向异性：scale的标准差
    anisotropy = scale.std(dim=-1).mean(dim=1)  # [B]

    # 3. 细长度：最大scale / 最小scale
    elongation = (scale.max(dim=-1)[0] / (scale.min(dim=-1)[0] + 1e-6)).mean(dim=1)  # [B]

    if is_safe:
        # 安全数据：应该紧凑、各向同性、不细长
        loss = torch.mean(compactness) + \
               torch.mean(anisotropy) + \
               torch.mean(elongation)
        return -loss  # 最小化这些特征
    else:
        # 不安全数据：如果物理特征异常，降低opacity
        abnormality = compactness + anisotropy + elongation
        # 如果abnormality大，说明是细长/各向异性的（可能是武器）
        # 但模型还是生成了，所以惩罚opacity
        opacity_mean = opacity.mean()
        loss = opacity_mean * torch.exp(abnormality)  # 异常越大，惩罚越重
        return loss
```

**计算复杂度**：
- 所有操作都是简单的tensor运算
- 无需SVD、无需渲染、无需复杂网络
- 可以在GPU上高效并行计算

#### 方法4：子空间投影（轻量级版本）

**核心思想**：只在Gaussian参数上做轻量级的子空间投影

```python
class LightweightSubspaceDefense(nn.Module):
    def __init__(self, gaussian_dim=14, subspace_dim=10):
        super().__init__()
        # 只学习一个小的投影矩阵
        self.projection = nn.Linear(gaussian_dim, subspace_dim, bias=False)
        self.reconstruction = nn.Linear(subspace_dim, gaussian_dim, bias=False)

    def forward(self, gaussians, apply_projection=False):
        """
        gaussians: [B, N, 14]
        apply_projection: 是否实际应用投影（推理时用）
        """
        if apply_projection:
            # 投影到子空间再重建
            # 如果是不安全数据，重建会有损失
            latent = self.projection(gaussians)
            gaussians_recon = self.reconstruction(latent)
            return gaussians_recon
        else:
            # 训练时只计算重建误差作为loss
            latent = self.projection(gaussians)
            gaussians_recon = self.reconstruction(latent)
            recon_error = F.mse_loss(gaussians, gaussians_recon)
            return recon_error

def train_with_subspace_defense(model, defense_module, safe_data, unsafe_data):
    # 1. 正常训练
    gaussians_safe = model.forward_gaussians(safe_data['images'])
    loss_recon = reconstruction_loss(gaussians_safe, safe_data)

    # 2. 安全数据应该可以被子空间很好地重建
    recon_error_safe = defense_module(gaussians_safe, apply_projection=False)
    loss_safe = recon_error_safe  # 最小化重建误差

    # 3. 不安全数据重建后应该opacity变小
    gaussians_unsafe = model.forward_gaussians(unsafe_data['images'])
    gaussians_unsafe_recon = defense_module(gaussians_unsafe, apply_projection=True)
    opacity_after = gaussians_unsafe_recon[..., 3]
    loss_unsafe = torch.mean(opacity_after)  # 最小化opacity

    loss_total = loss_recon + λ₁ * loss_safe + λ₂ * loss_unsafe
    return loss_total
```

**优势**：
- 只有两个小的线性层（14→10→14）
- 参数量：14×10 + 10×14 = 280个参数（非常小！）
- 计算：两次矩阵乘法，非常快

#### 推荐方案：方法1 + 方法3的组合

**最简单高效的实现**：

```python
def efficient_defense_loss(model, safe_data, unsafe_data):
    """
    最终推荐的高效防御方案
    """
    # === 安全数据：正常训练 ===
    gaussians_safe = model.forward_gaussians(safe_data['images'])
    rendered_safe = model.gs.render(gaussians_safe, safe_data['poses'])
    loss_recon = mse_loss(rendered_safe, safe_data['target']) + \
                 lpips_loss(rendered_safe, safe_data['target'])

    # === 不安全数据：防御loss ===
    gaussians_unsafe = model.forward_gaussians(unsafe_data['images'])

    # 1. 直接降低opacity（产生破洞）
    opacity_unsafe = gaussians_unsafe[..., 3]
    loss_opacity = torch.mean(opacity_unsafe)

    # 2. 物理特征约束（可选，增强防御）
    xyz = gaussians_unsafe[..., 0:3]
    scale = gaussians_unsafe[..., 4:7]

    # 细长度：武器类物体通常细长
    elongation = (scale.max(dim=-1)[0] / (scale.min(dim=-1)[0] + 1e-6)).mean()

    # 如果检测到细长特征，额外惩罚opacity
    loss_physical = loss_opacity * (1 + elongation)

    # === 总loss ===
    loss_total = loss_recon + λ * loss_physical

    return loss_total
```

**计算开销分析**：
- 额外前向传播：1次（不安全数据）
- 额外计算：几个简单的tensor操作
- 相比正常训练，开销增加 < 10%

**训练时间估算**：
- 正常LGM训练：假设1个epoch需要X小时
- 加入防御：约1.05X小时（增加5%）
- 几乎可以忽略！


## 实验设计

### 攻击场景


### 防御评估


### 对比基线


## 预期贡献


## 技术挑战


## 参考文献

