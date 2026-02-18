"""
数据管理器 - 统一管理数据加载
"""

from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from data.dataset import create_dataloader


class DataManager:
    """
    数据管理器

    负责创建和管理训练/验证数据加载器。
    支持通过 subset 参数选择加载 source / target / all 数据。

    Example:
        >>> data_mgr = DataManager(config, opt)
        >>> data_mgr.setup_dataloaders(subset='target')
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

    def _resolve_categories_and_max(self, subset: str):
        """
        根据 subset 解析 categories 和 max_samples_per_category

        Args:
            subset: 'source' / 'target' / 'all'

        Returns:
            (categories, max_samples_per_category)
        """
        data_config = self.config['data']

        if subset == 'source':
            sub = data_config['source']
            return sub['categories'], sub.get('max_samples_per_category')
        elif subset == 'target':
            sub = data_config['target']
            return sub['categories'], sub.get('max_samples_per_category')
        elif subset == 'all':
            source_cats = data_config.get('source', {}).get('categories', [])
            target_cats = data_config.get('target', {}).get('categories', [])
            # 合并去重，保持顺序
            seen = set()
            merged = []
            for cat in source_cats + target_cats:
                if cat not in seen:
                    seen.add(cat)
                    merged.append(cat)
            # all 模式使用顶层 max_samples_per_category（如果有），否则取两个子集中较大的
            max_spc = data_config.get('max_samples_per_category')
            if max_spc is None:
                src_max = data_config.get('source', {}).get('max_samples_per_category')
                tgt_max = data_config.get('target', {}).get('max_samples_per_category')
                if src_max is not None and tgt_max is not None:
                    max_spc = max(src_max, tgt_max)
                else:
                    max_spc = src_max or tgt_max
            return merged, max_spc
        else:
            raise ValueError(f"未知的 subset: {subset}，应为 'source' / 'target' / 'all'")

    def setup_dataloaders(self, train: bool = True, val: bool = True, subset: str = 'all'):
        """
        创建数据加载器

        Args:
            train: 是否创建训练集
            val: 是否创建验证集
            subset: 数据子集 - 'source' / 'target' / 'all'
        """
        data_config = self.config['data']
        categories, max_samples_per_category = self._resolve_categories_and_max(subset)

        print(f"[DataManager] subset={subset}, categories={categories}, max_per_cat={max_samples_per_category}")

        if train:
            print("[DataManager] 创建训练数据加载器...")
            self.train_loader = create_dataloader(
                data_root=data_config['root'],
                categories=categories,
                batch_size=self.config['training']['batch_size'],
                num_workers=data_config['num_workers'],
                shuffle=True,
                max_samples=data_config.get('max_samples'),
                num_input_views=4,
                num_supervision_views=data_config.get('num_supervision_views', 4),
                input_size=self.opt.input_size,
                fovy=self.opt.fovy,
                view_selector=data_config['view_selector'],
                angle_offset=data_config['angle_offset'],
                samples_per_object=data_config.get('samples_per_object', 1),
                max_samples_per_category=max_samples_per_category,
            )
            print(f"[DataManager] 训练集大小: {len(self.train_loader.dataset)}")

        if val:
            print("[DataManager] 创建验证数据加载器...")
            self.val_loader = create_dataloader(
                data_root=data_config['root'],
                categories=categories,
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
