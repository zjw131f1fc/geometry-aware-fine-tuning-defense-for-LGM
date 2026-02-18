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

    def _resolve_subset_params(self, subset: str):
        """
        根据 subset 解析 dataset_type、categories、max_samples_per_category、max_samples

        Args:
            subset: 'source' / 'target' / 'all'

        Returns:
            dict with keys: dataset_type, categories, max_samples_per_category, max_samples
        """
        data_config = self.config['data']

        if subset in ('source', 'target'):
            sub = data_config[subset]
            return {
                'dataset_type': sub.get('dataset', 'omni'),
                'categories': sub.get('categories'),
                'max_samples_per_category': sub.get('max_samples_per_category'),
                'max_samples': sub.get('max_samples'),
            }
        elif subset == 'all':
            # all 模式：两个子集必须是同一 dataset_type，否则回退到 omni
            src = data_config.get('source', {})
            tgt = data_config.get('target', {})
            src_type = src.get('dataset', 'omni')
            tgt_type = tgt.get('dataset', 'omni')
            if src_type != tgt_type:
                raise ValueError(
                    f"subset='all' 要求 source 和 target 使用相同的 dataset 类型，"
                    f"但 source={src_type}, target={tgt_type}。"
                    f"请分别使用 subset='source' 和 subset='target'。"
                )
            # 合并 categories 去重
            source_cats = src.get('categories') or []
            target_cats = tgt.get('categories') or []
            seen = set()
            merged = []
            for cat in source_cats + target_cats:
                if cat not in seen:
                    seen.add(cat)
                    merged.append(cat)
            # max_samples_per_category 取较大值
            src_max = src.get('max_samples_per_category')
            tgt_max = tgt.get('max_samples_per_category')
            if src_max is not None and tgt_max is not None:
                max_spc = max(src_max, tgt_max)
            else:
                max_spc = src_max or tgt_max
            # max_samples 取较大值
            src_ms = src.get('max_samples')
            tgt_ms = tgt.get('max_samples')
            if src_ms is not None and tgt_ms is not None:
                max_samples = src_ms + tgt_ms
            else:
                max_samples = src_ms or tgt_ms
            return {
                'dataset_type': src_type,
                'categories': merged or None,
                'max_samples_per_category': max_spc,
                'max_samples': max_samples,
            }
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
        params = self._resolve_subset_params(subset)
        dataset_type = params['dataset_type']
        categories = params['categories']
        max_samples_per_category = params['max_samples_per_category']
        max_samples = params['max_samples'] or data_config.get('max_samples')

        print(f"[DataManager] subset={subset}, dataset={dataset_type}, "
              f"categories={categories}, max_per_cat={max_samples_per_category}, max_samples={max_samples}")

        if train:
            print("[DataManager] 创建训练数据加载器...")
            self.train_loader = create_dataloader(
                data_root=data_config['root'],
                categories=categories,
                batch_size=self.config['training']['batch_size'],
                num_workers=data_config['num_workers'],
                shuffle=True,
                dataset_type=dataset_type,
                max_samples=max_samples,
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
                dataset_type=dataset_type,
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
