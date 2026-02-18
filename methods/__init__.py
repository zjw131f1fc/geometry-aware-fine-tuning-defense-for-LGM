"""
防御方法模块
"""

from .trap_losses import ScaleAnisotropyLoss, PositionCollapseLoss

__all__ = ['ScaleAnisotropyLoss', 'PositionCollapseLoss']
