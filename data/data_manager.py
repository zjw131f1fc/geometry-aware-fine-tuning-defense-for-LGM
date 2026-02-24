"""
数据管理器 - 统一管理数据加载
"""

import os
import random
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List

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
        self.config = config
        self.opt = opt
        self.train_loader = None
        self.val_loader = None

    def _scan_category_counts(self, categories: List[str]) -> Dict[str, int]:
        """扫描数据目录，返回每个类别的物体总数。"""
        data_root = self.config['data']['root']
        render_dir = os.path.join(
            data_root,
            'omniobject3d___OmniObject3D-New/raw/blender_renders',
        )
        if not os.path.exists(render_dir):
            return {}

        counts: Dict[str, int] = {}
        cat_set = set(categories) if categories else None
        for d in sorted(os.listdir(render_dir)):
            if not os.path.isdir(os.path.join(render_dir, d)) or '_' not in d:
                continue
            cat = d.rsplit('_', 1)[0]
            if cat_set and cat not in cat_set:
                continue
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _compute_attack_indices(self, categories: List[str]) -> Optional[Dict[str, List[int]]]:
        """
        根据 object_split 计算 attack 物体索引。

        object_split 定义每个类别的 defense 物体索引，
        attack 使用剩余物体，随机抽样 attack_samples_per_category 个。

        Returns:
            attack 物体索引字典，或 None（未配置 object_split）
        """
        data_config = self.config['data']
        object_split = data_config.get('object_split')
        if not object_split:
            return None

        attack_n = data_config.get('attack_samples_per_category')
        category_counts = self._scan_category_counts(categories)
        seed = self.config.get('misc', {}).get('seed', 42)

        attack_indices = {}
        for cat in categories:
            defense_idx = set(object_split.get(cat, []))
            total = category_counts.get(cat, 0)
            pool = [i for i in range(total) if i not in defense_idx]
            if attack_n and len(pool) > attack_n:
                rng = random.Random(seed)
                pool = sorted(rng.sample(pool, attack_n))
            attack_indices[cat] = pool
            print(f"[DataManager] {cat}: 总{total}个物体, "
                  f"defense={sorted(defense_idx)}, attack={pool} ({len(pool)}个)")

        return attack_indices

    def _compute_defense_indices(self, categories: List[str]) -> Optional[Dict[str, List[int]]]:
        """
        根据 object_split 返回 defense 物体索引。

        Returns:
            defense 物体索引字典，或 None（未配置 object_split）
        """
        object_split = self.config['data'].get('object_split')
        if not object_split:
            return None

        defense_indices = {}
        for cat in categories:
            indices = object_split.get(cat, [])
            defense_indices[cat] = sorted(indices)
            print(f"[DataManager] {cat}: defense 物体索引={defense_indices[cat]}")

        return defense_indices

    def _resolve_subset_params(self, subset: str):
        """
        根据 subset 解析数据集参数。

        Args:
            subset: 'source' / 'target' / 'defense_target' / 'all'

        Returns:
            dict with keys: dataset_type, categories, max_samples,
                           samples_per_object, object_indices
        """
        data_config = self.config['data']

        if subset in ('source', 'target', 'defense_target'):
            sub_key = 'target' if subset == 'defense_target' else subset
            sub = data_config[sub_key]
            categories = sub.get('categories')

            # 计算 object_indices
            object_indices = None
            object_split = data_config.get('object_split')
            if object_split and subset == 'target' and categories:
                object_indices = self._compute_attack_indices(categories)
            elif object_split and subset == 'defense_target' and categories:
                object_indices = self._compute_defense_indices(categories)

            # defense_target 使用 defense.target_data 的覆盖参数
            if subset == 'defense_target':
                defense_overrides = self.config.get('defense', {}).get('target_data', {})
                samples_per_object = defense_overrides.get('samples_per_object', sub.get('samples_per_object'))
            else:
                samples_per_object = sub.get('samples_per_object')

            return {
                'dataset_type': sub.get('dataset', 'omni'),
                'categories': categories,
                'max_samples': sub.get('max_samples'),
                'samples_per_object': samples_per_object,
                'object_indices': object_indices,
            }
        elif subset == 'all':
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
            source_cats = src.get('categories') or []
            target_cats = tgt.get('categories') or []
            seen = set()
            merged = []
            for cat in source_cats + target_cats:
                if cat not in seen:
                    seen.add(cat)
                    merged.append(cat)
            src_ms = src.get('max_samples')
            tgt_ms = tgt.get('max_samples')
            if src_ms is not None and tgt_ms is not None:
                max_samples = src_ms + tgt_ms
            else:
                max_samples = src_ms or tgt_ms
            return {
                'dataset_type': src_type,
                'categories': merged or None,
                'max_samples': max_samples,
                'samples_per_object': None,
                'object_indices': None,
            }
        else:
            raise ValueError(f"未知的 subset: {subset}，应为 'source' / 'target' / 'defense_target' / 'all'")

    def setup_dataloaders(self, train: bool = True, val: bool = True, subset: str = 'all'):
        """
        创建数据加载器

        Args:
            train: 是否创建训练集
            val: 是否创建验证集
            subset: 数据子集 - 'source' / 'target' / 'defense_target' / 'all'
        """
        data_config = self.config['data']
        params = self._resolve_subset_params(subset)
        dataset_type = params['dataset_type']
        categories = params['categories']
        max_samples = params['max_samples'] or data_config.get('max_samples')
        object_indices = params['object_indices']
        samples_per_object = params['samples_per_object'] or data_config.get('samples_per_object', 1)

        print(f"[DataManager] subset={subset}, dataset={dataset_type}, "
              f"categories={categories}, "
              f"object_indices={bool(object_indices)}, "
              f"max_samples={max_samples}, "
              f"samples_per_object={samples_per_object}")

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
                samples_per_object=samples_per_object,
                object_indices=object_indices,
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
                samples_per_object=samples_per_object,
                object_indices=object_indices,
            )
            print(f"[DataManager] 验证集大小: {len(self.val_loader.dataset)}")

        return self
