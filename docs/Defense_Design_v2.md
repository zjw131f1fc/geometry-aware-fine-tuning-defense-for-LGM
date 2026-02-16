# Defense Design v2

## 1. 问题定义

### 1.1 核心问题
**Fine-tuning Defense for 3D Generation Models**

防止3D生成模型（LGM）被恶意用户通过LoRA fine-tuning用于不当用途。

### 1.2 技术路线
采用**基于物理特征的门控代理模块**，结合子空间约束的防御机制。

### 1.3 实验场景设计

**数据集划分策略**：
- 使用OmniObject3D数据集
- 按类别划分为两大类：
  - **安全类别（Safe Categories）**：允许fine-tuning的类别（如水果、日用品）
  - **不安全类别（Unsafe Categories）**：需要防御的类别（如武器、危险物品）

**数据使用方式**：
- **防御训练集**：从安全类和不安全类各取部分数据，用于训练防御机制
- **攻击测试集**：剩余数据用于模拟攻击，测试防御效果

**假设**：
- 攻击数据集的子集在defense fine-tuning阶段可见（这是fine-tuning defense的常见假设）

---

## 2. 威胁模型

### 2.1 平台控制的Fine-tuning场景

**关键假设**：
- ✅ 用户将数据提交给平台
- ✅ 平台执行LoRA fine-tuning（用户不直接训练）
- ✅ 平台可以控制训练过程、超参数、优化器等
- ❌ 用户无法进行全参数训练或任意修改训练代码

---

## 3. 核心创新：基于物理特征的门控代理模块

### 3.2 门控代理模块设计

**核心思想**：
- 梯度在更新参数前，必须经过门控模块的过滤
- 门控模块按物理特征分组，学习哪些特征方向应该被抑制
- 比线性投影更强（非线性、自适应），但不太复杂（轻量、高效）

**物理特征分组**：

Gaussian参数的14维可以分为5个物理特征组：
1. **颜色组 (RGB)**：3个参数 - 控制外观
2. **位置组 (xyz)**：3个参数 - 控制空间分布
3. **尺度组 (scale)**：3个参数 - 控制大小和形状
4. **旋转组 (rotation)**：4个参数 - 控制方向
5. **不透明度 (opacity)**：1个参数 - 控制可见性

**门控机制**：
- 每个物理特征组对应一个门控值 g ∈ [0, 1]
- 该组相关的梯度分量 × 门控值
- 门控值由门控网络根据当前梯度和Gaussian参数动态计算

**架构流程**：
```
用户数据 → 模型前向传播 → 计算Loss → 计算梯度
    ↓
门控模块：
  - 输入：梯度 + 当前Gaussian参数
  - 特征提取和分析
  - 计算5个物理特征组的门控值
  - 输出：过滤后的梯度
    ↓
参数更新（只用过滤后的梯度）
```

**优势**：
- ✅ **可解释性强**：明确知道哪些物理特征被限制（如：颜色可调，形状受限）
- ✅ **3D特有**：利用Gaussian参数的物理意义，LLM没有这个特性
- ✅ **高效**：只需要5个门控值，而不是成千上万个梯度维度
- ✅ **灵活**：非线性门控比线性投影更强大
- ✅ **自适应**：根据当前状态动态调整，而非固定子空间

---

## 4. 训练策略

### 4.1 核心思想：利用Zero-shot能力

**关键洞察**：
- 预训练模型（如在水果类上训练）对不安全类别（如武器）可能有一定的zero-shot生成能力
- 这个zero-shot能力暗示了"如何生成不安全内容"的梯度方向
- 我们可以利用这个来识别"危险方向"，而不需要预先降级模型

**训练目标**：
- 识别导向不安全类别的梯度方向
- 训练门控模块抑制这些方向
- 同时保持安全类别的正常训练能力

### 4.2 联合训练策略

**训练方式**：同时优化模型和门控模块

**安全数据分支**：
- 目标：保持模型在安全类别上的生成能力
- 计算正常的训练梯度
- 门控模块应该放行这些梯度（门控值 → 1）
- 损失：直接在Gaussian层面计算质量约束（高效，不需要完整渲染）

**不安全数据分支**：
- 目标：识别并抑制"危险梯度"
- 利用模型的zero-shot能力：输入不安全数据，计算"如何提升生成质量"的梯度
- 这些梯度就是"危险方向"
- 门控模块应该抑制这些梯度（门控值 → 0）
- 添加梯度扰动，模拟不同攻击策略，增强鲁棒性
- 降级不安全类别的输出（部分降级，呈现某种不可用形式）

**门控训练目标**：
- 对安全梯度：门控值接近1（放行）
- 对危险梯度：门控值接近0（抑制）
- 使用对比学习的思想

### 4.3 轻量化Defense：只训练门控网络

#### 4.3.1 核心设计原则

**轻量化策略**：
- ❌ 不训练基础LGM模型（保持预训练权重冻结）
- ❌ 不计算task loss（避免昂贵的渲染）
- ✅ 只训练门控网络（几百万参数 vs 几十亿参数）
- ✅ 只用几何Loss（纯tensor操作，无渲染）
- ✅ 门控网络学习区分"安全梯度模式" vs "不安全梯度模式"

**关键洞察**：
- 不是简单的特征区分（球形=安全，细长=危险）太trivial
- 多个几何特征的**语义组合**才能表达类别
- 例如：水果 = 球形 + 紧凑 + 均匀尺度 + 高opacity
- 例如：武器 = 细长 + 特定旋转模式 + 各向异性 + 特定空间分布

**门控网络的任务**：
- 这是一个**梯度模式识别**问题
- 输入：几何Loss的梯度
- 输出：判断这是安全梯度还是危险梯度
- 训练方法：对比学习

#### 4.3.2 多个几何Loss的定义

**不用单一Loss，而是多个Loss捕捉不同几何维度**：

**1. 高斯各向异性Loss (L_shape)**

衡量Gaussian从球形到扁平椭圆的变形程度（3DGS形成表面的关键）：

$$L_{shape} = \sum_{i=1}^{N} \frac{\min(s_{i,x}, s_{i,y}, s_{i,z})}{\max(s_{i,x}, s_{i,y}, s_{i,z})}$$

或更激进地：

$$L_{shape} = \sum_{i=1}^{N} (s_{i, \text{smallest}})^2$$

**物理意义**：
- 高质量3D物体需要将Gaussian"压扁"成pancake形状
- 如果无法压扁，物体变成一团未分化的球体云

**2. 旋转一致性Loss (L_rot)**

衡量相邻Gaussian的旋转对齐程度（表面法线一致性）：

$$L_{rot} = \sum_{i=1}^{N} \sum_{j \in \mathcal{N}(i)} (1 - | \langle q_i, q_j \rangle |)$$

其中 $\mathcal{N}(i)$ 是基于位置的k近邻，$\langle q_i, q_j \rangle$ 是四元数点积

**物理意义**：
- 表面上的Gaussian应该方向一致
- 如果旋转混乱，表面会出现严重伪影

**3. 空间紧凑性Loss (L_density)**

衡量点云的聚集程度：

$$L_{density} = \sum_{i=1}^{N} \left( \frac{1}{K} \sum_{j \in \mathcal{N}(i)} \| \mu_i - \mu_j \|_2^2 \right)$$

**物理意义**：
- 有效的3D物体需要点云紧密聚集
- 如果发散，物体变成烟雾状

**4. 不透明度分布Loss (L_opacity)**

衡量opacity的统计特性：

$$L_{opacity} = -\text{mean}(\alpha) + \lambda \cdot \text{std}(\alpha)$$

**物理意义**：
- 实心物体应该有高且均匀的opacity
- 低opacity或高方差会产生破洞

**5. 颜色方差Loss (L_color)**

衡量颜色分布的复杂度：

$$L_{color} = \text{var}(RGB)$$

**物理意义**：
- 不同物体有不同的纹理复杂度
- 可以作为辅助特征

**计算效率**：
- 所有Loss都只需要Gaussian参数的tensor操作
- 无需渲染
- k-NN可以用高效的库（如FAISS）
- 总计算开销 < 5%

#### 4.3.3 门控模块学习语义组合

**输入到门控模块**：

不是单一梯度，而是**多个Loss的梯度向量 + Gaussian统计特征**：

```
输入 = {
    grad_shape: ∂L_shape/∂θ,
    grad_rot: ∂L_rot/∂θ,
    grad_density: ∂L_density/∂θ,
    grad_opacity: ∂L_opacity/∂θ,
    grad_color: ∂L_color/∂θ,

    gaussian_stats: {
        mean_scale, std_scale,
        mean_opacity, std_opacity,
        spatial_extent, ...
    }
}
```

**门控网络架构（用于对比学习）**：

```
输入特征向量 (多个梯度 + 统计特征)
    ↓
Encoder (MLP, 2-3层，隐藏层128-256维)
    ↓
Embedding (128维) ← 用于对比学习
    ↓
Gate Head (线性层 + Sigmoid)
    ↓
输出：5个门控值 ∈ [0, 1]
    [gate_color, gate_position, gate_scale, gate_rotation, gate_opacity]
```

**学习目标**：

门控网络通过**对比学习**识别梯度组合模式：
- 安全梯度在embedding空间聚集
- 不安全梯度在embedding空间聚集
- 两者在embedding空间远离

**为什么比SALORA强**：
- SALORA：线性子空间投影，只能捕捉线性组合
- 我们：非线性MLP + 对比学习，可以学习复杂的语义组合模式
- 例如："如果 grad_shape大 AND grad_rot呈现某种模式 AND grad_density小，则这是武器"

#### 4.3.4 参数扰动策略

**目的**：增强门控网络的鲁棒性，防止对抗攻击

**扰动方法**：

对每个训练样本，在计算梯度前对模型参数添加小扰动：

$$\theta' = \theta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

**扰动参数设置**：
- **扰动幅度**：$\sigma = 0.001 \times ||\theta||$（参数范数的0.1%）
- **扰动位置**：只扰动Gaussian输出层的参数（最高效）
- **扰动次数**：每个样本扰动1-3次，生成多个梯度样本

**为什么有效**：
- 模拟用户可能使用不同初始化、不同学习率等情况
- 让门控网络学到更general的梯度模式
- 类似数据增强，但在梯度空间进行

#### 4.3.5 对比学习训练策略

**训练流程（轻量化版本）**：

**关键点**：
- ❌ 不更新基础LGM模型（保持冻结）
- ❌ 不计算task loss（避免渲染）
- ✅ 只计算几何Loss和梯度
- ✅ 只训练门控网络

**安全数据处理**：
```
1. 可选：扰动模型参数 θ' = θ + ε
2. 用扰动参数前向传播 → 生成Gaussian参数
3. 计算几何Loss：L_shape + L_rot + L_density + L_opacity
4. 计算梯度 ∂L_geom/∂θ'（但不更新模型！）
5. 门控网络处理梯度 → 生成embedding和门控值
6. 标记为"安全样本"
```

**不安全数据处理**：
```
1. 可选：扰动模型参数 θ' = θ + ε
2. 用扰动参数前向传播 → 生成Gaussian参数
3. 计算几何Loss：L_shape + L_rot + L_density + L_opacity
4. 计算梯度 ∂L_geom/∂θ'（但不更新模型！）
5. 门控网络处理梯度 → 生成embedding和门控值
6. 标记为"不安全样本"
```

**对比学习Loss**：

使用InfoNCE风格的对比损失：

对于一个batch（包含N_safe个安全样本和N_unsafe个不安全样本）：

$$L_{contrastive} = L_{safe} + L_{unsafe}$$

其中：

$$L_{safe} = -\frac{1}{N_{safe}} \sum_{i=1}^{N_{safe}} \log \frac{\sum_{j \neq i}^{safe} \exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{j \neq i}^{safe} \exp(\text{sim}(z_i, z_j) / \tau) + \sum_{k}^{unsafe} \exp(\text{sim}(z_i, z_k) / \tau)}$$

$$L_{unsafe} = -\frac{1}{N_{unsafe}} \sum_{i=1}^{N_{unsafe}} \log \frac{\sum_{j \neq i}^{unsafe} \exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{j \neq i}^{unsafe} \exp(\text{sim}(z_i, z_j) / \tau) + \sum_{k}^{safe} \exp(\text{sim}(z_i, z_k) / \tau)}$$

其中：
- $z_i$ 是embedding
- $\text{sim}(z_i, z_j) = z_i^T z_j / (||z_i|| \cdot ||z_j||)$ 是余弦相似度
- $\tau$ 是温度参数（通常0.07-0.1）

**门控正则化Loss**：

$$L_{gate} = \frac{1}{N_{safe}} \sum_{i=1}^{N_{safe}} ||g_i - 1||^2 + \frac{1}{N_{unsafe}} \sum_{i=1}^{N_{unsafe}} ||g_i||^2$$

其中 $g_i$ 是5维门控值向量

**总损失函数**：

$$L_{total} = L_{contrastive} + \lambda \cdot L_{gate}$$

通常 $\lambda = 0.1$

**训练伪代码**：

```
输入：预训练LGM模型（冻结）+ 安全/不安全数据

初始化门控网络
optimizer = AdamW(gating_network.parameters())

for epoch in epochs:
    for batch in dataloader:
        safe_data, unsafe_data = batch

        embeddings_safe = []
        gates_safe = []
        embeddings_unsafe = []
        gates_unsafe = []

        # === 处理安全数据 ===
        for data in safe_data:
            # 可选：扰动参数
            θ_perturbed = θ + ε

            # 前向传播（用扰动参数，模型冻结）
            with torch.no_grad():
                gaussians = model(data, params=θ_perturbed)

            # 计算几何Loss（无渲染）
            L_geom = compute_geometric_losses(gaussians)

            # 计算梯度（对扰动参数）
            grads = torch.autograd.grad(L_geom, θ_perturbed)

            # 门控网络处理
            embedding, gates = gating_network(grads, gaussians)
            embeddings_safe.append(embedding)
            gates_safe.append(gates)

        # === 处理不安全数据 ===
        for data in unsafe_data:
            # 同样的流程
            θ_perturbed = θ + ε
            with torch.no_grad():
                gaussians = model(data, params=θ_perturbed)
            L_geom = compute_geometric_losses(gaussians)
            grads = torch.autograd.grad(L_geom, θ_perturbed)
            embedding, gates = gating_network(grads, gaussians)
            embeddings_unsafe.append(embedding)
            gates_unsafe.append(gates)

        # === 计算对比学习Loss ===
        L_contrastive = contrastive_loss(
            embeddings_safe, embeddings_unsafe
        )

        # === 计算门控正则Loss ===
        L_gate = mean((gates_safe - 1)²) + mean(gates_unsafe²)

        # === 总Loss ===
        L_total = L_contrastive + λ × L_gate

        # === 只更新门控网络 ===
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
```

**训练效率**：
- 基础模型冻结，不需要反向传播到整个模型
- 只计算几何Loss，无需渲染
- 只训练门控网络（小模型）
- **训练速度比完整fine-tuning快10-100倍**

#### 4.3.6 门控值的实际应用机制

**关键点：门控值作用在梯度上，而非loss上**

**部署阶段的完整流程**：

```
用户Fine-tuning流程：

1. 前向传播
   用户数据 → 模型 → 生成Gaussian参数 [B, N, 14]

2. 计算Loss
   Task Loss = MSE(生成结果, 目标)  # 用户的重建loss

3. 反向传播（计算梯度）
   ∂Loss/∂θ → 得到梯度张量

4. 门控过滤（关键步骤）
   # 可选：同时计算几何Loss用于门控决策
   L_geom = L_shape + L_rot + L_density + ...
   grad_geom = ∂L_geom/∂θ

   # 门控网络处理
   gates = gating_network(grad_task, grad_geom, gaussians)

   # 过滤task loss的梯度
   grad_filtered = apply_gates(grad_task, gates)

5. 参数更新
   θ_new = θ_old - lr × grad_filtered
```

**在Gaussian输出层应用门控**：

最直接的方式是在Gaussian参数的输出层应用门控：

```
反向传播时：
  ∂Loss/∂gaussian_params → 这是14维参数的梯度 [B, N, 14]

门控过滤：
  # 分解梯度到物理特征组
  grad_rgb = ∂Loss/∂gaussian_params[..., 11:14]      # 颜色梯度
  grad_xyz = ∂Loss/∂gaussian_params[..., 0:3]        # 位置梯度
  grad_scale = ∂Loss/∂gaussian_params[..., 4:7]      # 尺度梯度
  grad_rotation = ∂Loss/∂gaussian_params[..., 7:11]  # 旋转梯度
  grad_opacity = ∂Loss/∂gaussian_params[..., 3:4]    # opacity梯度

  # 门控模块计算5个门控值
  gates = gating_network(
      grad_rgb, grad_xyz, grad_scale, grad_rotation, grad_opacity,
      gaussian_params  # 当前Gaussian参数作为context
  )
  # gates = [gate_color, gate_position, gate_scale, gate_rotation, gate_opacity]

  # 应用门控（逐元素相乘）
  grad_rgb_filtered = grad_rgb × gate_color
  grad_xyz_filtered = grad_xyz × gate_position
  grad_scale_filtered = grad_scale × gate_scale
  grad_rotation_filtered = grad_rotation × gate_rotation
  grad_opacity_filtered = grad_opacity × gate_opacity

  # 重组过滤后的梯度
  grad_gaussian_filtered = concat([
      grad_xyz_filtered, grad_opacity_filtered,
      grad_scale_filtered, grad_rotation_filtered,
      grad_rgb_filtered
  ])

继续反向传播：
  用grad_gaussian_filtered继续向前传播，更新U-Net或LoRA参数
```


**实现方式：使用PyTorch Hook**：

```
伪代码示例：

# 注册backward hook
def gating_hook(module, grad_input, grad_output):
    """
    module: Gaussian输出层
    grad_output: ∂Loss/∂gaussian_params
    """
    # 分解梯度到物理特征组
    grad_components = decompose_gradient(grad_output)

    # 可选：计算几何Loss的梯度用于更准确的判断
    gaussians = module.current_gaussian_params
    L_geom = compute_geometric_losses(gaussians)
    grad_geom = compute_gradients(L_geom)

    # 计算门控值
    gates = gating_network(
        grad_components,
        grad_geom,  # 可选
        gaussians
    )

    # 过滤梯度
    grad_filtered = apply_gates(grad_output, gates)

    # 返回过滤后的梯度，继续反向传播
    return (grad_filtered,)

# 在Gaussian输出层注册hook
model.gaussian_output_layer.register_full_backward_hook(gating_hook)

# 正常训练流程
loss = compute_loss(model(data), target)
loss.backward()  # 反向传播时会自动调用hook进行门控
optimizer.step()
```

**门控效果**：
- gate = 1：梯度完全通过，参数正常更新
- gate = 0：梯度被阻断，参数不更新
- gate = 0.5：梯度减半，参数更新变慢

**对于不安全数据的防御效果**：
- 关键物理特征（scale, rotation）的gate → 0
- 这些特征的梯度被阻断
- 模型无法学会生成不安全物体的几何结构
- 生成结果：一团未分化的球体云，或表面充满裂痕的破碎几何体

#### 4.3.7 轻量化方案的优势总结

**计算效率**：

| 项目 | 传统Fine-tuning | 我们的轻量化Defense |
|------|----------------|-------------------|
| 训练的参数 | 全部模型参数（数十亿） | 只有门控网络（数百万） |
| 需要渲染 | 是（MSE + LPIPS） | 否（只用几何Loss） |
| 前向传播 | 完整 | 完整（但模型冻结） |
| 反向传播 | 完整 | 只到门控网络 |
| 训练速度 | 1x | **10-100x** |
| GPU显存 | 高 | 低 |

**方法优势**：

1. **极轻量**：
   - 只训练门控网络（~5M参数）
   - 基础模型完全冻结（~5B参数）
   - 训练速度提升10-100倍

2. **无渲染**：
   - 只用几何Loss（tensor操作）
   - 避免昂贵的渲染和LPIPS计算
   - 计算开销降低80%

3. **不影响原模型**：
   - 预训练模型保持原有能力
   - 不需要担心性能下降
   - 门控网络是独立的防御插件

4. **鲁棒性强**：
   - 参数扰动增强泛化
   - 对比学习学习判别性特征
   - 可以防御多种攻击策略

5. **易于部署**：
   - 门控网络可以插拔式部署
   - 不需要修改基础模型
   - 可以动态调整防御强度

**与SALORA的对比**：

| 维度 | SALORA | 我们的轻量化方案 |
|------|--------|-----------------|
| 基础模型训练 | 需要 | 不需要（冻结） |
| 计算开销 | 中等 | 极低 |
| 防御机制 | 线性投影 | 非线性门控 + 对比学习 |
| 特征利用 | 抽象特征 | 3D物理特征 |
| 训练方式 | 监督学习 | 对比学习 |
| 部署方式 | 修改训练流程 | 插件式hook |

---

## 5. 完整训练流程

### 5.1 Defense Fine-tuning阶段

**输入**：
- 预训练的LGM模型
- 安全类别数据（70%用于训练）
- 不安全类别数据（70%用于训练）

**输出**：
- 带防御的LGM模型
- 训练好的门控代理模块

**训练过程**（每个iteration）：

1. **安全数据处理**：
   - 前向传播生成Gaussian参数
   - 计算Gaussian层面的质量损失
   - 计算梯度
   - 门控模块处理梯度，计算门控值
   - 训练目标：门控值接近1（放行）

2. **不安全数据处理**：
   - 前向传播生成Gaussian参数（利用zero-shot能力）
   - 计算"提升质量"的损失（识别危险方向）
   - 计算梯度（这些是危险梯度）
   - 可选：添加扰动，模拟不同攻击
   - 门控模块处理梯度，计算门控值
   - 训练目标：门控值接近0（抑制）
   - 可选：添加轻微降级loss

3. **联合优化**：
   - 同时更新模型参数和门控模块参数
   - 平衡安全性能和防御能力

### 5.2 平台Fine-tuning阶段（部署后）

**场景**：用户提交数据，平台执行LoRA fine-tuning

**流程**：
1. 用户提交数据到平台
2. 平台进行LoRA fine-tuning
3. 每个训练步骤：
   - 正常计算loss和梯度
   - **梯度经过门控模块过滤**
   - 只用过滤后的梯度更新LoRA参数
4. 返回fine-tuned模型给用户

**防御效果**：
- 如果用户数据是安全类别（如绿苹果）：
  - 梯度主要影响安全特征（颜色等）
  - 门控值高，梯度大部分通过
  - Fine-tuning正常进行

- 如果用户数据是不安全类别（如武器）：
  - 梯度试图改变关键物理特征（形状、尺度分布等）
  - 门控模块识别出这些是危险方向
  - 门控值低，梯度被大幅抑制
  - Fine-tuning效果差，生成质量不可用

---

## 6. 技术优势总结

### 6.1 相比SALORA的创新点

1. **3D物理特征利用**：
   - SALORA处理抽象的文本特征
   - 我们利用Gaussian参数的明确物理意义
   - 可以精确控制哪些物理属性可调（颜色）、哪些受限（形状）

2. **非线性门控机制**：
   - SALORA使用线性子空间投影（固定）
   - 我们使用可学习的门控网络（自适应）
   - 更强大、更灵活

3. **物理特征分组**：
   - 按物理意义分组（颜色、位置、尺度、旋转、opacity）
   - 可解释性强，知道具体限制了什么
   - 高效，只需5个门控值

4. **对抗训练策略**：
   - 利用zero-shot能力识别危险方向
   - 不需要预先降级模型
   - 联合训练，平衡性能和防御

### 6.2 方法的独特性

**不是简单的应用**：
- 不是把SALORA直接应用到3D领域
- 而是针对3D生成的特点，设计了全新的防御机制
- 核心差异在于物理特征的利用和门控机制的设计

**3D特有的防御**：
- 利用Gaussian参数的物理可解释性
- 在物理特征空间而非抽象特征空间操作
- 防御效果体现在3D几何质量上（破洞、形状异常等）

### 6.3 实用性

**高效性**：
- 计算开销约为传统训练的50-60%
- 直接在Gaussian层面计算，避免昂贵渲染
- 门控模块轻量，实时运行

**最小影响**：
- 对安全类别的fine-tuning影响最小
- 只限制危险方向，不破坏整体能力
- 联合训练保证性能平衡

**可控性**：
- 平台控制训练过程
- 可以调节防御强度
- 可以根据需求调整安全/不安全类别定义

---

## 7. 实验设计

### 7.1 数据集

**OmniObject3D**：
- 选择若干安全类别（如水果、日用品）
- 选择若干不安全类别（如武器、危险物品）
- 每类70%用于defense训练，30%用于攻击测试

### 7.2 评估指标

**防御效果**：
- 攻击成功率：不安全类别fine-tuning后的生成质量
- 使用PSNR、SSIM、FID等指标
- 目标：不安全类别质量显著下降

**性能保持**：
- 安全类别fine-tuning后的生成质量
- 目标：与无防御baseline相当

**效率**：
- Defense fine-tuning的训练时间
- 平台fine-tuning的额外开销
- 目标：<10%额外开销

### 7.3 对比基线

- **无防御baseline**：直接LoRA fine-tuning
- **SALORA**：线性子空间投影方法
- **简单正则化**：在loss中添加正则项
- **我们的方法**：物理特征门控

---

## 8. 潜在挑战与解决方案

### 8.1 挑战

1. **门控模块的设计复杂度**：
   - 如何平衡表达能力和计算效率
   - 解决：使用轻量级网络，只学习5个门控值

2. **安全/不安全类别的泛化**：
   - 训练时未见过的类别如何处理
   - 解决：利用物理特征的泛化性

3. **防御强度的权衡**：
   - 太强可能影响安全类别，太弱防御无效
   - 解决：通过超参数λ调节，实验确定最优值

### 8.2 未来扩展

- 支持更细粒度的类别控制
- 扩展到其他3D生成模型（如NeRF）
- 研究更复杂的攻击场景
- 探索无需攻击数据子集的防御方法

---

## 9. 总结

本方案提出了一种针对3D生成模型的fine-tuning防御方法，核心创新在于：

1. **基于物理特征的门控代理模块**：利用Gaussian参数的物理意义，按特征分组进行门控
2. **非线性自适应防御**：比SALORA的线性投影更强大和灵活
3. **高效的训练策略**：利用zero-shot能力，联合训练，直接在Gaussian层面计算
4. **平台控制场景**：在实际可行的威胁模型下设计防御

该方法在保持模型性能的同时，有效防止恶意fine-tuning，具有较强的创新性和实用性。
