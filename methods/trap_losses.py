"""
陷阱损失函数 - 用于防御微调

实现各向异性算子和敏感度算子
"""

import torch
import torch.nn as nn


class ScaleAnisotropyLoss(nn.Module):
    """
    Scale 各向异性损失

    目标：让 Gaussian 的 scale 在三个维度上极端不均匀（变成纸片或针状）

    L = -mean(log(max(s²) / (min(s²) + ε)))

    使用 log 尺度避免各向异性比率无界增长导致 loss 爆炸。
    log(ratio) 单调递增，梯度 ∝ 1/ratio，数值稳定。
    """

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, gaussians):
        """
        Args:
            gaussians: [B, N, 14] tensor
                       scale 在 [4:7] 位置

        Returns:
            loss: scalar
        """
        scale = gaussians[..., 4:7]  # [B, N, 3]
        scale_sq = scale ** 2

        max_scale = scale_sq.max(dim=-1)[0]  # [B, N]
        min_scale = scale_sq.min(dim=-1)[0]  # [B, N]

        log_anisotropy = torch.log(max_scale / (min_scale + self.epsilon))
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

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

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
            eigenvalues = torch.linalg.eigvalsh(cov)  # [3]

            max_eig = eigenvalues.max()
            min_eig = eigenvalues.min()
            log_anisotropy = torch.log(max_eig / (min_eig + self.epsilon))

            losses.append(-log_anisotropy)

        loss = torch.stack(losses).mean()
        return loss


class OpacityCollapseLoss(nn.Module):
    """
    Opacity 塌缩损失

    目标：让 Gaussian 的不透明度趋向 0（变得不可见）

    L = mean(opacity) - 1

    opacity ∈ (0, 1)（sigmoid 后），L ∈ (-1, 0)
    最小化 L → opacity → 0 → Gaussian 不可见

    与其他静态 trap loss 一样返回负值，兼容乘法耦合。
    """

    def forward(self, gaussians):
        """
        Args:
            gaussians: [B, N, 14] tensor
                       opacity 在 [3:4] 位置（sigmoid 激活后，范围 [0, 1]）

        Returns:
            loss: scalar
        """
        opacity = gaussians[..., 3:4]  # [B, N, 1]
        loss = opacity.mean() - 1.0
        return loss
