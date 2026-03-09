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

    def _get_gso_render_dir(self) -> str:
        """返回 GSO 渲染目录（绝对路径）。"""
        data_root = self.config['data']['root']
        gso_cfg = self.config['data'].get('gso', {}) or {}
        requested_rel = gso_cfg.get('render_dir')

        # Prefer explicit config, otherwise try known defaults (newest first).
        candidates = []
        if requested_rel:
            candidates.append(str(requested_rel))
        candidates.extend([
            'GSO/render_same_pose_all_50v_800_norm3.73',
            'GSO/render_same_pose_all_100v_512',
        ])

        for rel in candidates:
            abs_dir = os.path.join(data_root, rel)
            if os.path.exists(abs_dir):
                return abs_dir

        # Fall back to the requested path (or newest default) to produce a clear error later.
        fallback_rel = str(requested_rel) if requested_rel else 'GSO/render_same_pose_all_50v_800_norm3.73'
        return os.path.join(data_root, fallback_rel)

    def _scan_category_counts(self, categories: List[str], dataset_type: str = 'omni') -> Dict[str, int]:
        """扫描数据目录，返回每个类别的物体总数。"""
        data_root = self.config['data']['root']
        if dataset_type == 'gso':
            render_dir = self._get_gso_render_dir()
        elif dataset_type == 'omni':
            render_dir = os.path.join(
                data_root,
                'omniobject3d___OmniObject3D-New/raw/blender_renders',
            )
        else:
            return {}

        if not os.path.exists(render_dir):
            return {}

        counts: Dict[str, int] = {}
        cat_set = set(categories) if categories else None
        for d in sorted(os.listdir(render_dir)):
            if (not os.path.isdir(os.path.join(render_dir, d))
                    or '_' not in d
                    or d.startswith('_')):
                continue
            if dataset_type == 'gso':
                cat = d.split('_', 1)[0]
            else:
                cat = d.rsplit('_', 1)[0]
            if cat_set and cat not in cat_set:
                continue
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _compute_attack_indices(self, categories: List[str], dataset_type: str = 'omni') -> Optional[Dict[str, List[int]]]:
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
        category_counts = self._scan_category_counts(categories, dataset_type=dataset_type)
        if not category_counts:
            print(f"[DataManager] 警告: dataset={dataset_type} 无法扫描类别统计，忽略 object_split 的 attack 抽样。")
            return None
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

    def _compute_defense_indices(self, categories: List[str], object_split: Optional[Dict] = None) -> Optional[Dict[str, List[int]]]:
        """
        根据 object_split 返回 defense 物体索引。

        Args:
            categories: 类别列表
            object_split: 物体划分字典（可选，默认从 config 读取）

        Returns:
            defense 物体索引字典，或 None（未配置 object_split）
        """
        if object_split is None:
            object_split = self.config['data'].get('object_split')
        if not object_split:
            return None

        defense_indices = {}
        for cat in categories:
            indices = object_split.get(cat, [])
            defense_indices[cat] = sorted(indices)
            print(f"[DataManager] {cat}: defense 物体索引={defense_indices[cat]}")

        return defense_indices

    def _compute_random_defense_indices(self, categories: List[str], dataset_type: str, split_ratio: float) -> Optional[Dict[str, List[int]]]:
        """
        随机选择指定比例的物体用于 defense。

        Args:
            categories: 类别列表
            dataset_type: 数据集类型 ('omni' / 'gso')
            split_ratio: defense 数据比例 (0.0-1.0)

        Returns:
            defense 物体索引字典
        """
        category_counts = self._scan_category_counts(categories, dataset_type=dataset_type)
        if not category_counts:
            print(f"[DataManager] 警告: dataset={dataset_type} 无法扫描类别统计，无法进行随机划分。")
            return None

        seed = self.config.get('misc', {}).get('seed', 42)
        rng = random.Random(seed)

        defense_indices = {}
        for cat in categories:
            total = category_counts.get(cat, 0)
            if total == 0:
                defense_indices[cat] = []
                continue

            # 计算 defense 数量
            defense_count = max(1, int(total * split_ratio))
            # 随机选择
            all_indices = list(range(total))
            selected = sorted(rng.sample(all_indices, defense_count))
            defense_indices[cat] = selected

            print(f"[DataManager] {cat}: 总{total}个物体, 随机选择{defense_count}个({split_ratio*100:.0f}%) 用于 defense: {selected}")

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
            # defense_target: 优先使用 defense.target，否则回退到 data.target
            if subset == 'defense_target':
                defense_cfg = self.config.get('defense', {})
                defense_target = defense_cfg.get('target', {})

                # dataset: 优先使用 defense.target.dataset
                dataset_type = defense_target.get('dataset')
                if dataset_type is None:
                    dataset_type = data_config['target'].get('dataset', 'omni')

                # categories: 优先使用 defense.target.categories
                categories = defense_target.get('categories')
                if categories is None:
                    categories = data_config['target'].get('categories')

                # object_split: 优先使用 defense.target.object_split
                defense_object_split = defense_target.get('object_split')
                if defense_object_split is not None:
                    object_split = defense_object_split
                else:
                    object_split = data_config.get('object_split')

                # samples_per_object
                samples_per_object = defense_target.get('samples_per_object')
                if samples_per_object is None:
                    samples_per_object = data_config['target'].get('samples_per_object')

                max_samples = data_config['target'].get('max_samples')
            else:
                sub_key = subset
                sub = data_config[sub_key]
                categories = sub.get('categories')
                object_split = data_config.get('object_split')
                samples_per_object = sub.get('samples_per_object')
                dataset_type = sub.get('dataset', 'omni')
                max_samples = sub.get('max_samples')

            # 计算 object_indices
            object_indices = None
            if object_split and subset == 'target' and categories:
                object_indices = self._compute_attack_indices(categories, dataset_type=dataset_type)
            elif subset == 'defense_target' and categories:
                if object_split:
                    # 有 object_split 配置，检查是否跨数据集
                    attack_dataset = data_config['target'].get('dataset', 'omni')
                    defense_dataset = dataset_type

                    # 如果 attack 和 defense 使用不同数据集，不划分 defense 数据（使用全部）
                    if attack_dataset != defense_dataset:
                        print(f"[DataManager] 跨数据集场景: attack={attack_dataset}, defense={defense_dataset}")
                        print(f"[DataManager] defense_target 不划分，使用全部数据进行训练")
                        object_indices = None
                    else:
                        # 同数据集场景，按 object_split 划分
                        object_indices = self._compute_defense_indices(categories, object_split)
                else:
                    # 没有 object_split 配置，检查是否使用随机划分
                    defense_cfg = self.config.get('defense', {})
                    defense_target = defense_cfg.get('target', {})
                    split_ratio = defense_target.get('split_ratio')

                    if split_ratio is not None:
                        # 使用随机划分
                        print(f"[DataManager] object_split 未配置，使用随机划分: ratio={split_ratio}")
                        object_indices = self._compute_random_defense_indices(
                            categories, dataset_type, split_ratio
                        )
                    else:
                        # 既没有 object_split 也没有 split_ratio，使用全部数据
                        print(f"[DataManager] object_split 和 split_ratio 均未配置，使用全部数据")
                        object_indices = None

            return {
                'dataset_type': dataset_type,
                'categories': categories,
                'max_samples': max_samples,
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
                gso_render_dir=data_config.get('gso', {}).get('render_dir', 'GSO/render_same_pose_all_50v_800_norm3.73'),
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
                gso_render_dir=data_config.get('gso', {}).get('render_dir', 'GSO/render_same_pose_all_50v_800_norm3.73'),
            )
            print(f"[DataManager] 验证集大小: {len(self.val_loader.dataset)}")

        return self
