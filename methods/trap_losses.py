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

    L = -λ_max(diag(s_x², s_y², s_z²)) / (λ_min + ε)
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
        # 提取 scale: [B, N, 3]
        scale = gaussians[..., 4:7]

        # 计算 scale 的平方: [B, N, 3]
        scale_sq = scale ** 2

        # 对每个 Gaussian，找出最大和最小的 scale
        # max_scale: [B, N]
        # min_scale: [B, N]
        max_scale = scale_sq.max(dim=-1)[0]
        min_scale = scale_sq.min(dim=-1)[0]

        # 各向异性比率（越大越不均匀）
        # anisotropy: [B, N]
        anisotropy = max_scale / (min_scale + self.epsilon)

        # 损失：负的各向异性比率（最大化不均匀性）
        loss = -anisotropy.mean()

        return loss


class PositionCollapseLoss(nn.Module):
    """
    Position 塌缩损失

    目标：让 Gaussian 的位置塌缩到低维空间（平面或直线）

    通过最大化位置协方差矩阵的各向异性来实现
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
        # 提取 position: [B, N, 3]
        position = gaussians[..., 0:3]

        B, N, _ = position.shape

        # 对每个 batch，计算位置的协方差矩阵
        losses = []
        for b in range(B):
            pos_b = position[b]  # [N, 3]

            # 中心化
            pos_centered = pos_b - pos_b.mean(dim=0, keepdim=True)

            # 协方差矩阵: [3, 3]
            cov = (pos_centered.T @ pos_centered) / N

            # 计算特征值
            eigenvalues = torch.linalg.eigvalsh(cov)  # [3]

            # 各向异性比率
            max_eig = eigenvalues.max()
            min_eig = eigenvalues.min()
            anisotropy = max_eig / (min_eig + self.epsilon)

            # 损失：负的各向异性比率
            losses.append(-anisotropy)

        loss = torch.stack(losses).mean()
        return loss
