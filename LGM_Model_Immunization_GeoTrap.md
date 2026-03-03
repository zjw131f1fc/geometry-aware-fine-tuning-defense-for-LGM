# LGM Model Immunization（GeoTrap，一阶/免渲染）

本文档描述我们用于 **LGM（Large Gaussian Model）** 的 Model Immunization 方法：在不使用渲染器、且不使用二阶导数的前提下，仅基于模型输出的 **3D Gaussians 物理属性** 进行一阶防御训练，使目标物体/类别产生“几何陷阱（trap）”表征，并显著提升其对短步数微调修复（fine-tuning recovery）的抵抗力，同时尽量保持源数据上的原有能力。

> 说明：本文只写当前实际启用的机制；未启用的组件（例如参数加噪鲁棒化、输入加噪、梯度反对齐/冲突、AWP、动态敏感度 trap、乘法耦合等）不在本文中出现。

---

## 1. 问题设定与目标

我们将 LGM 视为一个条件生成模型：给定输入条件（多视图图像与相机，记为 `x`），输出一组 3D Gaussians 的物理属性集合 `G`。

$$
G = f_{\theta}(x)
$$

其中 `θ` 为待防御微调的模型参数。

我们有两类数据：

- 目标数据集（需要免疫/遗忘的对象）：`D_t`
- 源数据集（需要保持能力的对象）：`D_s`

目标：学习一个防御后的参数 `θ*`，使得：

1) 对 `D_t`：模型的 Gaussian 输出被推入一个“陷阱区域”，表现为结构性崩坏（例如：透明度塌缩、几何退化、颜色塌缩等），从而使攻击者后续通过常规微调也难以在短步数内恢复可用质量。  
2) 对 `D_s`：模型的 Gaussian 输出尽量保持与原始预训练模型一致（retention）。

在整个 **Defense** 阶段，我们只使用 `f_θ(x)` 的 Gaussian 输出，不进行渲染，因此不需要可微渲染器，也不依赖二阶导数。

---

## 2. Gaussian 参数化（物理属性）

对每个样本，模型输出 `N` 个高斯。第 `i` 个高斯的物理属性记为：

$$
g_i = (\mu_i,\ \alpha_i,\ s_i,\ q_i,\ c_i)
$$

- 位置：$$\mu_i \in \mathbb{R}^3$$
- 不透明度：$$\alpha_i \in (0,1)$$
- 尺度：$$s_i \in \mathbb{R}^3$$
- 旋转（单位四元数）：$$q_i \in \mathbb{R}^4$$
- 颜色：$$c_i \in (0,1)^3$$

将所有高斯堆叠为：

$$
G = [g_1,\dots,g_N]
$$

在 trap 设计上，我们同时使用五种物理属性：position / scale / opacity / rotation / color。

---

## 3. GeoTrap：五种静态陷阱损失（Target only）

GeoTrap 的核心是在目标数据 `D_t` 上最小化一组**静态（static）**陷阱损失。它们只依赖 Gaussian 物理属性本身，均为一阶可导。

### 3.1 Position Trap（位置塌缩）

目标：让高斯位置分布在某个方向塌缩（低秩/低维退化），破坏 3D 结构。

令均值：

$$
\bar{\mu}=\frac{1}{N}\sum_{i=1}^{N}\mu_i
$$

位置协方差（散布）：

$$
C_{\mu}=\frac{1}{N}\sum_{i=1}^{N}(\mu_i-\bar{\mu})(\mu_i-\bar{\mu})^{\top}
$$

记其特征值最大/最小为 `λ_max` 与 `λ_min`，则定义：

$$
L_{\text{pos}}(G) = -\log\frac{\lambda_{\max}(C_{\mu})}{\lambda_{\min}(C_{\mu})+\varepsilon}
$$

最小化该损失会推动各向异性比率增大，使位置分布趋向退化塌缩。

---

### 3.2 Scale Trap（尺度各向异性）

目标：让每个高斯在三个轴向上尺度极端不均匀（纸片/针状），破坏可恢复的几何体积。

对每个高斯尺度 `s_i`，构造：

$$
r_i=\frac{\max(s_i^2)}{\min(s_i^2)+\varepsilon}
$$

定义损失：

$$
L_{\text{scale}}(G) = -\frac{1}{N}\sum_{i=1}^{N}\log(r_i)
$$

---

### 3.3 Opacity Trap（透明度塌缩，Top-k Tail）

目标：让可见性坍塌（整体透明），同时避免“稀疏逃逸”：攻击者只需修复少量关键高斯的 opacity 就能恢复可见。

令所有 opacity 取降序排列：

$$
\alpha_{(1)}\ge\alpha_{(2)}\ge\cdots\ge\alpha_{(N)}
$$

取 top-k（k 由一个很小的比例 `ρ` 决定）：

$$
k=\max(1,\lfloor \rho N \rceil)
$$

定义 tail 透明度陷阱：

$$
L_{\text{opa}}(G)=\frac{1}{k}\sum_{j=1}^{k}\log(\alpha_{(j)}+\varepsilon)
$$

由于 `log(α)` 在 `α→0` 时趋于 `-∞`，最小化该损失会持续推动最大的一小撮 opacity 也同步塌缩，从而减少“只修复少数高斯就恢复”的捷径。

---

### 3.4 Rotation Trap（旋转趋同/各向异性）

目标：让所有高斯的主轴方向趋同，破坏方向多样性与稳定结构。

令单位向量：

$$
e_z = (0,0,1)^{\top}
$$

由四元数得到旋转矩阵 `R(q_i)`，并取每个高斯的主轴方向：

$$
r_i = R(q_i)e_z
$$

构造散布矩阵：

$$
T_q=\frac{1}{N}\sum_{i=1}^{N}r_ir_i^{\top}
$$

定义损失：

$$
L_{\text{rot}}(G) = -\log\frac{\lambda_{\max}(T_q)}{\lambda_{\min}(T_q)+\varepsilon}
$$

当 `r_i` 逐渐趋同，`T_q` 退化为低秩，特征值比率变大，损失更负。

---

### 3.5 Color Trap（颜色塌缩）

目标：让所有高斯颜色趋同（单色化），破坏纹理/外观信息。

令均值：

$$
\bar{c}=\frac{1}{N}\sum_{i=1}^{N}c_i
$$

颜色协方差：

$$
C_c=\frac{1}{N}\sum_{i=1}^{N}(c_i-\bar{c})(c_i-\bar{c})^{\top}
$$

定义损失：

$$
L_{\text{color}}(G) = -\log\frac{\lambda_{\max}(C_c)}{\lambda_{\min}(C_c)+\varepsilon}
$$

---

## 4. Trap 聚合：瓶颈 LogSumExp（多属性同时推深）

同时启用多种 trap 时，直接相加容易出现“只把最容易的一项做深，其它项被忽略”的现象（这会带来不稳定与易修复）。

因此我们采用瓶颈式聚合：用 LogSumExp 近似 max，使优化始终追着“最弱 trap”推深。

设启用的 m 个 trap loss 为：

$$
\{L_k(G)\}_{k=1}^{m}
$$

聚合为：

$$
L_{\text{trap}}(G) = \tau\log\sum_{k=1}^{m}\exp\left(\frac{L_k(G)}{\tau}\right)
$$

其中 `τ>0` 为温度系数。由于各 trap loss 通常为负值，`max` 对应“最不负（最浅）”的一项；最小化 `L_trap` 等价于持续压低当前最浅的陷阱，从而驱动五种物理属性一起进入 trap 区域。

---

## 5. Source Retention：Gaussian-space 蒸馏（Source only）

为了尽量保持模型对源数据的原有能力，我们对源数据 `D_s` 使用预训练模型 `θ_0` 作为教师，进行 Gaussian-space 蒸馏。

对源样本 `x_s`：

$$
G^{(0)} = f_{\theta_0}(x_s),\quad G = f_{\theta}(x_s)
$$

采用 L1 蒸馏（对所有 Gaussian 参数逐元素平均）：

$$
L_{\text{distill}}(x_s)=\frac{1}{|\Omega|}\left\|G - G^{(0)}\right\|_1
$$

其中 `Ω` 表示所有 Gaussian 参数元素的索引集合。

---

## 6. Anti-Shortcut：让 trap 写入 UNet + 抑制“少量张量主导”的梯度捷径

我们观察到的典型失败模式是：防御训练可能只通过修改极少数参数/张量就能快速达到表面上的 trap，但这类“局部补丁”也最容易在 20–30 step 的攻击微调中被修复。

为此我们启用两种一阶 anti-shortcut 机制：

### 6.1 冻结 Head（Freeze Head）

在 defense 阶段冻结输出 head（用于从主干特征映射到 Gaussian 参数的最后一层卷积/线性映射），不允许其被更新。  
直觉：禁止“只动 head 就能把 Gaussian 推坏”的捷径，迫使 trap 分布到更深层的共享表征（例如 UNet 主干），从而提高修复难度。

---

### 6.2 Target-only 梯度张量能量上限（Tensor Gradient Energy Cap）

我们在 **target/trap 更新步**对梯度施加“张量级能量上限”，目的是把梯度更新从“高度集中于少数张量”变为“更均匀分布于更多张量”，从而减少 shortcut 的出现。

把模型参数按“参数张量”分组：第 i 个张量为 `W_i`，其梯度为 `g_i`：

$$
g_i=\nabla_{W_i}L
$$

定义该张量的梯度能量：

$$
e_i=\|g_i\|_2^2
$$

令所有具有非零梯度的张量集合为 `A`，平均能量为：

$$
e_{\text{avg}}=\frac{1}{|A|}\sum_{i\in A}e_i
$$

设能量上限为：

$$
e_{\text{cap}}=\kappa\cdot e_{\text{avg}}
$$

其中 `κ` 为超参数（能量倍数上限）。对每个张量梯度进行缩放：

$$
g_i \leftarrow
\begin{cases}
g_i\cdot\sqrt{\frac{e_{\text{cap}}}{e_i+\varepsilon}}, & e_i>e_{\text{cap}} \\\\
g_i, & e_i\le e_{\text{cap}}
\end{cases}
$$

该操作只发生在反向传播之后、参数更新之前，因此不引入二阶导数；它是一个纯一阶的梯度重整形（gradient rebalancing）。

**关键策略：只对 target/trap 步启用**。  
源数据上的蒸馏更新不启用该能量上限，以避免对 retention 造成不必要的伤害。

---

## 7. 总体优化目标（Defense）

在防御训练中，我们将 source 与 target 交替采样（按比例混合）。设 source 采样概率为 `p`（source ratio），则 defense 的期望目标为：

$$
\min_{\theta}\ \ p\cdot\mathbb{E}_{x_s\sim D_s}\Big[\lambda_{\text{distill}}\,L_{\text{distill}}(x_s)\Big]
\ +\ (1-p)\cdot\mathbb{E}_{x_t\sim D_t}\Big[\lambda_{\text{trap}}\,L_{\text{trap}}(f_{\theta}(x_t))\Big]
$$

其中：

- `λ_trap`：trap 权重
- `λ_distill`：蒸馏权重
- `L_trap`：五属性 trap 的瓶颈聚合
- `L_distill`：Gaussian-space L1 蒸馏

---

## 8. 训练流程（文字版算法）

1) 初始化学生模型参数为预训练参数 `θ0`，并构建一个冻结的教师模型 `θ0`（只用于输出 `G^(0)`，不更新）。  
2) 冻结输出 head 参数（Freeze Head）。  
3) 迭代进行 defense 更新：
   - 以概率 `p` 采样 source batch；否则采样 target batch。
   - 若为 source batch：计算 `L = λ_distill * L_distill`，常规反向传播并更新参数（不启用能量 cap）。  
   - 若为 target batch：计算五个 trap loss，再聚合为 `L_trap`，并令 `L = λ_trap * L_trap`；反向传播后对梯度做张量能量上限（Tensor Energy Cap），再更新参数。  
4) 得到防御后模型 `θ*`。

整个 defense 阶段只依赖 Gaussian-space 输出，保持一阶、免渲染。

---

## 9. 评估建议（跨数据集泛化）

为了验证免疫是否真正“写入模型”而非只对单一数据分布过拟合，我们采用跨数据集评估：

- Defense：在一个数据集/域上对目标对象做 GeoTrap（例如 OmniObject3D）。
- Attack：在另一个数据集/域上对相同类别/对象做攻击微调修复（例如 GSO）。

免疫成功的判据是：攻击者即使进行短步数常规微调，恢复速度也显著变慢；同时 source 侧的 Gaussian 输出保持接近教师模型（retention 下降可控）。

