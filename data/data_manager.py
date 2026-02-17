"""
数据管理器 - 统一管理数据加载
"""

from torch.utils.data import DataLoader
from typing import Dict, Any

from data.dataset import create_dataloader


class DataManager:
    """
    数据管理器

    负责创建和管理训练/验证数据加载器

    Example:
        >>> data_mgr = DataManager(config, opt)
        >>> data_mgr.setup_dataloaders()
        >>> train_loader = data_mgr.train_loader
    """

    def __init__(self, config: Dict[str, Any], opt):
        """
        初始化数据管理器

        Args:
            config: 配置字典
            opt: LGM 配置选项
        """
        self.config = config
        self.opt = opt
        self.train_loader = None
        self.val_loader = None

    def setup_dataloaders(self, train: bool = True, val: bool = True):
        """
        创建数据加载器

        Args:
            train: 是否创建训练集
            val: 是否创建验证集
        """
        data_config = self.config['data']

        if train:
            print("[DataManager] 创建训练数据加载器...")
            self.train_loader = create_dataloader(
                data_root=data_config['root'],
                categories=data_config['categories'],
                batch_size=self.config['training']['batch_size'],
                num_workers=data_config['num_workers'],
                shuffle=True,
                max_samples=data_config['max_samples'],
                num_input_views=4,
                num_supervision_views=data_config.get('num_supervision_views', 4),
                input_size=self.opt.input_size,
                fovy=self.opt.fovy,
                view_selector=data_config['view_selector'],
                angle_offset=data_config['angle_offset'],
                samples_per_object=data_config.get('samples_per_object', 1),
                max_samples_per_category=data_config.get('max_samples_per_category', None),
            )
            print(f"[DataManager] 训练集大小: {len(self.train_loader.dataset)}")

        if val:
            print("[DataManager] 创建验证数据加载器...")
            self.val_loader = create_dataloader(
                data_root=data_config['root'],
                categories=data_config['categories'],
                batch_size=self.config['training']['batch_size'],
                num_workers=data_config['num_workers'],
                shuffle=False,
                max_samples=data_config.get('val_samples', 20),
                num_input_views=4,
                num_supervision_views=data_config.get('num_supervision_views', 4),
                input_size=self.opt.input_size,
                fovy=self.opt.fovy,
                view_selector=data_config['view_selector'],
                angle_offset=data_config['angle_offset'],
                samples_per_object=1,
                max_samples_per_category=None,
            )
            print(f"[DataManager] 验证集大小: {len(self.val_loader.dataset)}")

        return self
