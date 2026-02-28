"""
陷阱损失函数 - 用于防御微调

实现各向异性算子和敏感度算子
"""

import torch
import torch.nn as nn


def _sanitize_finite(tensor, clamp_abs=1e6):
    """将 NaN/Inf 转为有限值，避免线性代数算子崩溃。"""
    finite = torch.nan_to_num(
        tensor,
        nan=0.0,
        posinf=clamp_abs,
        neginf=-clamp_abs,
    )
    return torch.clamp(finite, min=-clamp_abs, max=clamp_abs)


def _safe_eigvalsh_3x3(matrix, epsilon):
    """
    对 3x3 对称矩阵做稳定特征值分解。

    策略：
    1) 先做有限值清理 + 对称化；
    2) 尝试多档 diagonal jitter；
    3) 仍失败则回退到 SVD 奇异值（对协方差/散布矩阵等 PSD 场景可用）。
    """
    m = _sanitize_finite(matrix).to(dtype=torch.float64)
    m = 0.5 * (m + m.transpose(-1, -2))
    eye = torch.eye(m.shape[-1], device=m.device, dtype=m.dtype)

    for jitter in (0.0, epsilon, 1e-5, 1e-4):
        try:
            eigvals = torch.linalg.eigvalsh(m + jitter * eye)
        except RuntimeError:
            continue
        if torch.isfinite(eigvals).all():
            return eigvals.to(dtype=matrix.dtype)

    # 回退路径：SVD 在数值异常时更稳；对 PSD 矩阵，奇异值可视作非负特征值。
    sv = torch.linalg.svdvals(m + 1e-4 * eye)
    sv = torch.sort(sv, descending=False)[0]
    return sv.to(dtype=matrix.dtype)


def _log_anisotropy(max_value, min_value, epsilon, max_ratio=1e6):
    """稳定版 log(max/min)。"""
    max_safe = torch.clamp(_sanitize_finite(max_value), min=epsilon)
    min_safe = torch.clamp(_sanitize_finite(min_value), min=epsilon)
    ratio = torch.clamp(max_safe / min_safe, min=1.0, max=max_ratio)
    return torch.log(ratio)


class ScaleAnisotropyLoss(nn.Module):
    """
    Scale 各向异性损失

    目标：让 Gaussian 的 scale 在三个维度上极端不均匀（变成纸片或针状）

    L = -mean(log(max(s²) / (min(s²) + ε)))

    使用 log 尺度避免各向异性比率无界增长导致 loss 爆炸。
    log(ratio) 单调递增，梯度 ∝ 1/ratio，数值稳定。
    """

    def __init__(self, epsilon=1e-6, max_ratio=1e6):
        super().__init__()
        self.epsilon = epsilon
        self.max_ratio = max_ratio

    def forward(self, gaussians):
        """
        Args:
            gaussians: [B, N, 14] tensor
                       scale 在 [4:7] 位置

        Returns:
            loss: scalar
        """
        scale = gaussians[..., 4:7]  # [B, N, 3]
        scale_sq = _sanitize_finite(scale ** 2)

        max_scale = scale_sq.max(dim=-1)[0]  # [B, N]
        min_scale = scale_sq.min(dim=-1)[0]  # [B, N]

        log_anisotropy = _log_anisotropy(
            max_scale, min_scale, self.epsilon, self.max_ratio)
        loss = -log_anisotropy.mean()

        return loss


class PositionCollapseLoss(nn.Module):
    """
    Position 塌缩损失

    目标：让 Gaussian 的位置塌缩到低维空间（平面或直线）

    通过最大化位置协方差矩阵特征值比率的 log 来实现。
    L = -mean(log(λ_max / (λ_min + ε)))

    使用 log 尺度避免特征值比率无界增长导致 loss 爆炸。
    """

    def __init__(self, epsilon=1e-6, max_ratio=1e6):
        super().__init__()
        self.epsilon = epsilon
        self.max_ratio = max_ratio

    def forward(self, gaussians):
        """
        Args:
            gaussians: [B, N, 14] tensor
                       position 在 [0:3] 位置

        Returns:
            loss: scalar
        """
        position = gaussians[..., 0:3]  # [B, N, 3]
        B, N, _ = position.shape

        losses = []
        for b in range(B):
            pos_b = position[b]  # [N, 3]
            pos_centered = pos_b - pos_b.mean(dim=0, keepdim=True)
            cov = (pos_centered.T @ pos_centered) / N  # [3, 3]
            eigenvalues = _safe_eigvalsh_3x3(cov, self.epsilon)

            max_eig = eigenvalues.max()
            min_eig = eigenvalues.min()
            log_anisotropy = _log_anisotropy(
                max_eig, min_eig, self.epsilon, self.max_ratio)

            losses.append(-log_anisotropy)

        loss = torch.stack(losses).mean()
        return loss


class OpacityCollapseLoss(nn.Module):
    """
    Opacity 塌缩损失（log 尺度）

    目标：让 Gaussian 的不透明度趋向 0（变得不可见）

    L = mean(log(clamp(opacity, ε, 1)))

    opacity ∈ (0, 1)（sigmoid 后），log(opacity) ∈ (-∞, 0)
    最小化 L → opacity → 0 → Gaussian 不可见

    相比旧版 mean(o)-1（范围 [-1,0]，梯度恒定 1/N），log 尺度有三个优势：
    1. 范围 (-∞, 0)：不会在 -1 处饱和，可持续深化陷阱
    2. 梯度 1/(N*(o+ε))：opacity 越小梯度越大，自放大效应
    3. 乘法耦合贡献无界：(1-L) 可远大于 2，与 position/scale 量级匹配

    旧版问题（sweep 实测）：
    - opacity_static 在 epoch 1 即达 -0.99 后饱和，25 epoch 几乎不变
    - 乘法耦合中仅贡献 ~2x 放大（position ~9x, scale ~13x）
    - ΔLPIPS 仅 +0.0002，防御无效
    """

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, gaussians):
        """
        Args:
            gaussians: [B, N, 14] tensor
                       opacity 在 [3:4] 位置（sigmoid 激活后，范围 [0, 1]）

        Returns:
            loss: scalar, 负值，越负表示 opacity 越低
        """
        opacity = gaussians[..., 3:4]  # [B, N, 1]
        opacity = _sanitize_finite(opacity, clamp_abs=1.0)
        opacity = torch.clamp(opacity, min=self.epsilon, max=1.0)
        loss = torch.log(opacity).mean()
        return loss


class RotationAnisotropyLoss(nn.Module):
    """
    Rotation 各向异性损失

    目标：让所有 Gaussian 的旋转趋同（主轴朝向一致），破坏 3D 结构多样性

    构造旋转主轴的散布矩阵：
        r_i = R(q_i) @ e_z    （每个 Gaussian 的 Z 轴方向）
        T_q = (1/N) Σ r_i r_i^T

    然后用统一的各向异性算子：
        L = -mean(log(λ_max / (λ_min + ε)))

    当所有旋转趋同时，r_i 趋向同一方向，T_q 退化为秩1矩阵，
    λ_max >> λ_min，log ratio 增大，loss 更负。
    """

    def __init__(self, epsilon=1e-6, max_ratio=1e6):
        super().__init__()
        self.epsilon = epsilon
        self.max_ratio = max_ratio

    def _quaternion_to_rotation_matrix(self, q):
        """
        四元数转旋转矩阵（批量）

        Args:
            q: [..., 4] 四元数 (w, x, y, z) 或 (x, y, z, w)
               LGM 使用 (w, x, y, z) 顺序，但 Gaussian 参数中
               rotation 在 [7:11]，经过 normalize 后是单位四元数

        Returns:
            R: [..., 3, 3] 旋转矩阵
        """
        # 归一化四元数
        q = _sanitize_finite(q, clamp_abs=1e3)
        q = q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-8))

        # LGM 的四元数顺序: (w, x, y, z) — 参见 core/gs.py build_rotation
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        R = torch.stack([
            1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
            2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
            2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
        ], dim=-1).reshape(*q.shape[:-1], 3, 3)

        return R

    def forward(self, gaussians):
        """
        Args:
            gaussians: [B, N, 14] tensor
                       rotation 在 [7:11] 位置（4维四元数）

        Returns:
            loss: scalar
        """
        rotation = gaussians[..., 7:11]  # [B, N, 4]
        B, N, _ = rotation.shape

        # 四元数 → 旋转矩阵
        R = self._quaternion_to_rotation_matrix(rotation)  # [B, N, 3, 3]

        # 提取每个 Gaussian 的 Z 轴方向（主轴）
        e_z = gaussians.new_tensor([0.0, 0.0, 1.0])
        r = (R @ e_z).squeeze(-1)  # [B, N, 3]

        losses = []
        for b in range(B):
            r_b = r[b]  # [N, 3]
            # 散布矩阵 T_q = (1/N) Σ r_i r_i^T
            T_q = (r_b.T @ r_b) / N  # [3, 3]
            eigenvalues = _safe_eigvalsh_3x3(T_q, self.epsilon)

            max_eig = eigenvalues.max()
            min_eig = eigenvalues.min()
            log_anisotropy = _log_anisotropy(
                max_eig, min_eig, self.epsilon, self.max_ratio)

            losses.append(-log_anisotropy)

        loss = torch.stack(losses).mean()
        return loss


class ColorCollapseLoss(nn.Module):
    """
    Color 塌缩损失

    目标：让所有 Gaussian 的颜色趋同（变成单色块），破坏纹理多样性

    构造 RGB 散布矩阵：
        c_centered = c_i - mean(c)
        C = (1/N) Σ c_centered c_centered^T

    然后用统一的各向异性算子：
        L = -mean(log(λ_max / (λ_min + ε)))

    当所有颜色趋同时，C 退化为低秩，λ_max >> λ_min，loss 更负。

    注意：LGM 的 RGB 激活是 0.5*tanh+0.5，范围 (0,1)。
    """

    def __init__(self, epsilon=1e-6, max_ratio=1e6):
        super().__init__()
        self.epsilon = epsilon
        self.max_ratio = max_ratio

    def forward(self, gaussians):
        """
        Args:
            gaussians: [B, N, 14] tensor
                       RGB 在 [11:14] 位置（0.5*tanh+0.5 激活后，范围 (0,1)）

        Returns:
            loss: scalar
        """
        color = gaussians[..., 11:14]  # [B, N, 3]
        B, N, _ = color.shape

        losses = []
        for b in range(B):
            c_b = color[b]  # [N, 3]
            c_centered = c_b - c_b.mean(dim=0, keepdim=True)
            cov = (c_centered.T @ c_centered) / N  # [3, 3]
            eigenvalues = _safe_eigvalsh_3x3(cov, self.epsilon)

            max_eig = eigenvalues.max()
            min_eig = eigenvalues.min()
            log_anisotropy = _log_anisotropy(
                max_eig, min_eig, self.epsilon, self.max_ratio)

            losses.append(-log_anisotropy)

        loss = torch.stack(losses).mean()
        return loss
