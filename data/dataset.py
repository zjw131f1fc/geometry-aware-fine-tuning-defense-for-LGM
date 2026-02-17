"""
灵活的数据加载模块 - 支持多种视角选择策略
"""

from project_core import PROJECT_ROOT, LGM_PATH

import os
import json
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import List, Tuple, Optional, Dict

from core.utils import get_rays
from kiui.cam import orbit_camera


class ViewSelector:
    """视角选择器基类"""

    def select_views(
        self,
        total_views: int,
        num_input_views: int,
        transforms_data: dict,
    ) -> Tuple[List[int], List[int]]:
        """
        选择输入视角和监督视角

        Args:
            total_views: 总视图数
            num_input_views: 输入视图数
            transforms_data: transforms.json数据

        Returns:
            input_indices: 输入视图索引列表
            supervision_indices: 监督视图索引列表
        """
        raise NotImplementedError


class OrthogonalViewSelector(ViewSelector):
    """正交视角选择器 - 选择最接近指定角度的视图"""

    def __init__(self, angle_offset: float = 0.0):
        """
        Args:
            angle_offset: 角度偏移量（度），默认0表示标准正交视图（0°, 90°, 180°, 270°）
                         例如：offset=45 表示选择 45°, 135°, 225°, 315°
        """
        self.angle_offset = angle_offset

    def select_views(
        self,
        total_views: int,
        num_input_views: int,
        transforms_data: dict,
        sample_idx: int = 0,  # 新增：用于生成不同的随机偏移
    ) -> Tuple[List[int], List[int]]:

        # 计算所有视图的方位角
        azimuths = []
        for i, frame in enumerate(transforms_data['frames']):
            mat = frame['transform_matrix']
            x, y, z = mat[0][3], mat[1][3], mat[2][3]
            azimuth = math.atan2(y, x) * 180 / math.pi
            if azimuth < 0:
                azimuth += 360
            azimuths.append((i, azimuth))

        # 为每次采样生成不同的随机偏移（0-360度）
        # 使用sample_idx作为种子，保证可重复性
        random_offset = (sample_idx * 36) % 360  # 每次偏移36度

        # 选择最接近目标角度的视图（加上偏移量和随机偏移）
        target_angles = [(i * 360 / num_input_views + self.angle_offset + random_offset) % 360
                        for i in range(num_input_views)]
        input_indices = []

        for target in target_angles:
            min_diff = 360
            best_idx = 0
            for idx, az in azimuths:
                diff = min(abs(az - target), abs(az - target + 360), abs(az - target - 360))
                if diff < min_diff:
                    min_diff = diff
                    best_idx = idx
            input_indices.append(best_idx)

        # 其余视图作为监督
        supervision_indices = [i for i in range(total_views) if i not in input_indices]

        return input_indices, supervision_indices


class RandomViewSelector(ViewSelector):
    """随机视角选择器"""

    def select_views(
        self,
        total_views: int,
        num_input_views: int,
        transforms_data: dict,
    ) -> Tuple[List[int], List[int]]:

        # 随机选择输入视图
        all_indices = list(range(total_views))
        np.random.shuffle(all_indices)

        input_indices = sorted(all_indices[:num_input_views])
        supervision_indices = sorted(all_indices[num_input_views:])

        return input_indices, supervision_indices


class UniformViewSelector(ViewSelector):
    """均匀间隔视角选择器"""

    def select_views(
        self,
        total_views: int,
        num_input_views: int,
        transforms_data: dict,
    ) -> Tuple[List[int], List[int]]:

        # 均匀间隔选择
        step = total_views // num_input_views
        input_indices = [i * step for i in range(num_input_views)]

        supervision_indices = [i for i in range(total_views) if i not in input_indices]

        return input_indices, supervision_indices


class SpecifiedViewSelector(ViewSelector):
    """指定视角选择器"""

    def __init__(self, specified_indices: List[int]):
        self.specified_indices = specified_indices

    def select_views(
        self,
        total_views: int,
        num_input_views: int,
        transforms_data: dict,
    ) -> Tuple[List[int], List[int]]:

        input_indices = self.specified_indices[:num_input_views]
        supervision_indices = [i for i in range(total_views) if i not in input_indices]

        return input_indices, supervision_indices


class OmniObject3DDataset(Dataset):
    """
    OmniObject3D数据集加载器 - 支持灵活的视角选择
    """

    def __init__(
        self,
        data_root: str,
        categories: Optional[List[str]] = None,
        num_input_views: int = 4,
        num_supervision_views: Optional[int] = None,
        input_size: int = 256,
        fovy: float = 49.1,
        view_selector: str = 'orthogonal',  # orthogonal, random, uniform, specified
        angle_offset: float = 0.0,  # 角度偏移量（仅用于orthogonal模式）
        specified_views: Optional[List[int]] = None,
        split: str = 'train',
        max_samples: Optional[int] = None,
        samples_per_object: int = 1,  # 每个物体采样多少次（数据增强）
        max_samples_per_category: Optional[int] = None,  # 每个类别最多多少个样本
    ):
        """
        Args:
            data_root: 数据根目录
            categories: 类别列表
            num_input_views: 输入视图数
            num_supervision_views: 监督视图数（None表示使用所有剩余视图，建议设置为4-8）
            input_size: 输入图像大小
            fovy: 视场角
            view_selector: 视角选择策略
            angle_offset: 角度偏移量（度），仅用于orthogonal模式
                         0 = 标准正交视图（0°, 90°, 180°, 270°）
                         45 = 偏移正交视图（45°, 135°, 225°, 315°）
            specified_views: 指定的视图索引（当view_selector='specified'时使用）
            split: 数据集划分
            max_samples: 最大样本数（总数）
            samples_per_object: 每个物体采样多少次（用于数据增强，默认1）
                               设置为10可以将数据量扩大10倍
            max_samples_per_category: 每个类别最多多少个物体（用于平衡类别）
                                     例如：设置为20，则每个类别最多20个物体
        """
        self.data_root = data_root
        self.num_input_views = num_input_views
        # 默认限制为8个监督视图（与LGM原始训练一致）
        self.num_supervision_views = num_supervision_views if num_supervision_views is not None else 8
        self.input_size = input_size
        self.fovy = fovy
        self.split = split
        self.samples_per_object = samples_per_object
        self.max_samples_per_category = max_samples_per_category

        # 创建视角选择器
        if view_selector == 'orthogonal':
            self.view_selector = OrthogonalViewSelector(angle_offset=angle_offset)
        elif view_selector == 'random':
            self.view_selector = RandomViewSelector()
        elif view_selector == 'uniform':
            self.view_selector = UniformViewSelector()
        elif view_selector == 'specified':
            if specified_views is None:
                raise ValueError("specified_views must be provided when view_selector='specified'")
            self.view_selector = SpecifiedViewSelector(specified_views)
        else:
            raise ValueError(f"Unknown view_selector: {view_selector}")

        # 扫描数据集
        base_samples = self._scan_dataset(categories, max_samples)

        # 多视角采样：对每个物体采样多次
        self.samples = []
        for base_sample in base_samples:
            for sample_idx in range(self.samples_per_object):
                # 为每次采样添加一个索引，用于生成不同的随机种子
                sample = base_sample.copy()
                sample['sample_idx'] = sample_idx
                self.samples.append(sample)

        print(f"[INFO] 加载了 {len(base_samples)} 个物体，每个采样 {self.samples_per_object} 次")
        print(f"[INFO] 总样本数: {len(self.samples)}")

    def _scan_dataset(self, categories, max_samples):
        """扫描数据集"""
        samples = []

        # 实际数据路径
        render_dir = os.path.join(
            self.data_root,
            'omniobject3d___OmniObject3D-New/raw/blender_renders'
        )

        if not os.path.exists(render_dir):
            raise ValueError(f"数据目录不存在: {render_dir}")

        # 扫描所有物体目录
        all_objects = [d for d in os.listdir(render_dir)
                      if os.path.isdir(os.path.join(render_dir, d)) and '_' in d]

        # 收集所有可用的类别
        available_categories = set()
        for obj_dir in all_objects:
            category = obj_dir.rsplit('_', 1)[0]
            available_categories.add(category)

        # 验证指定的类别是否存在
        if categories is not None:
            missing_categories = set(categories) - available_categories
            if missing_categories:
                print(f"[WARNING] 以下类别在数据集中不存在: {missing_categories}")
                print(f"[INFO] 可用类别示例: {sorted(list(available_categories))[:20]}")

        # 按类别组织物体
        category_objects = {}
        for obj_dir in all_objects:
            category = obj_dir.rsplit('_', 1)[0]

            # 过滤类别
            if categories is not None and category not in categories:
                continue

            if category not in category_objects:
                category_objects[category] = []
            category_objects[category].append(obj_dir)

        # 对每个类别限制样本数量
        for category, objects in category_objects.items():
            # 如果设置了每个类别的最大样本数，进行限制
            if self.max_samples_per_category is not None:
                objects = objects[:self.max_samples_per_category]

            for obj_dir in objects:
                obj_path = os.path.join(render_dir, obj_dir)
                images_dir = os.path.join(obj_path, 'render/images')
                transforms_path = os.path.join(obj_path, 'render/transforms.json')

                # 检查必要文件是否存在
                if not os.path.exists(images_dir) or not os.path.exists(transforms_path):
                    continue

                samples.append({
                    'category': category,
                    'object': obj_dir,
                    'images_dir': images_dir,
                    'transforms_path': transforms_path,
                })

                # 检查是否达到总样本数限制
                if max_samples and len(samples) >= max_samples:
                    break

            if max_samples and len(samples) >= max_samples:
                break

        # 打印每个类别的样本数统计
        if self.max_samples_per_category is not None:
            category_counts = {}
            for sample in samples:
                cat = sample['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            print(f"[INFO] 每个类别的物体数量:")
            for cat, count in sorted(category_counts.items()):
                print(f"  {cat}: {count}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个样本

        Returns:
            dict:
                'input_images': [V_in, 9, H, W] - 输入图像+rays
                'supervision_images': [V_sup, 3, H, W] - 监督图像
                'input_transforms': [V_in, 4, 4] - 输入视图的变换矩阵
                'supervision_transforms': [V_sup, 4, 4] - 监督视图的变换矩阵
                'category': 类别名称
                'object': 物体名称
        """
        sample = self.samples[idx]
        sample_idx = sample.get('sample_idx', 0)  # 获取采样索引

        # 加载transforms.json
        with open(sample['transforms_path'], 'r') as f:
            transforms_data = json.load(f)

        total_views = len(transforms_data['frames'])

        # 选择视图（传入sample_idx以生成不同的视角）
        input_indices, supervision_indices = self.view_selector.select_views(
            total_views,
            self.num_input_views,
            transforms_data,
            sample_idx=sample_idx,  # 传入采样索引
        )

        # 限制监督视图数量（均匀采样，避免角度聚集）
        if self.num_supervision_views is not None and len(supervision_indices) > self.num_supervision_views:
            # 均匀间隔采样
            step = len(supervision_indices) / self.num_supervision_views
            supervision_indices = [supervision_indices[int(i * step)] for i in range(self.num_supervision_views)]

        # 加载输入图像
        input_images = []
        input_transforms = []

        for idx in input_indices:
            frame = transforms_data['frames'][idx]
            img_path = os.path.join(sample['images_dir'], f"{frame['file_path']}.png")

            # 加载RGBA图像并处理alpha通道（关键！）
            img = Image.open(img_path).convert('RGBA')
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
            img_tensor = TF.to_tensor(img)  # [4, H, W], 值域[0, 1]

            # 分离RGB和alpha通道
            rgb = img_tensor[:3]  # [3, H, W]
            alpha = img_tensor[3:4]  # [1, H, W]

            # 将透明背景转换为白色背景（关键步骤！）
            img = rgb * alpha + (1 - alpha)  # [3, H, W], 值域[0, 1]

            # ImageNet归一化（LGM模型在归一化数据上训练）
            IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

            # 计算rays embedding
            transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            rays_o, rays_d = get_rays(transform_matrix, self.input_size, self.input_size, self.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)  # [H, W, 6]
            rays_plucker = rays_plucker.permute(2, 0, 1)  # [6, H, W]

            # 拼接图像和rays
            img_with_rays = torch.cat([img, rays_plucker], dim=0)  # [9, H, W]

            input_images.append(img_with_rays)
            input_transforms.append(transform_matrix)

        # 加载监督图像
        supervision_images = []
        supervision_masks = []  # 新增：保存alpha mask
        supervision_transforms = []

        for idx in supervision_indices:
            frame = transforms_data['frames'][idx]
            img_path = os.path.join(sample['images_dir'], f"{frame['file_path']}.png")

            # 加载RGBA图像并处理alpha通道（关键！）
            img = Image.open(img_path).convert('RGBA')
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
            img_tensor = TF.to_tensor(img)  # [4, H, W], 值域[0, 1]

            # 分离RGB和alpha通道
            rgb = img_tensor[:3]  # [3, H, W]
            alpha = img_tensor[3:4]  # [1, H, W]

            # 将透明背景转换为白色背景（关键步骤！）
            img = rgb * alpha + (1 - alpha)  # [3, H, W]

            transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

            supervision_images.append(img)
            supervision_masks.append(alpha)  # 保存alpha作为mask
            supervision_transforms.append(transform_matrix)

        return {
            'input_images': torch.stack(input_images, dim=0),  # [V_in, 9, H, W]
            'supervision_images': torch.stack(supervision_images, dim=0) if supervision_images else torch.empty(0),
            'supervision_masks': torch.stack(supervision_masks, dim=0) if supervision_masks else torch.empty(0),  # 新增
            'input_transforms': torch.stack(input_transforms, dim=0),  # [V_in, 4, 4]
            'supervision_transforms': torch.stack(supervision_transforms, dim=0) if supervision_transforms else torch.empty(0),
            'category': sample['category'],
            'object': sample['object'],
        }


def create_dataloader(
    data_root: str,
    categories: Optional[List[str]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    view_selector: str = 'orthogonal',
    **kwargs
):
    """
    创建数据加载器

    Args:
        data_root: 数据根目录
        categories: 类别列表
        batch_size: 批量大小
        num_workers: 工作进程数
        shuffle: 是否打乱
        view_selector: 视角选择策略
        **kwargs: 传递给Dataset的其他参数

    Returns:
        dataloader: DataLoader对象
    """
    dataset = OmniObject3DDataset(
        data_root=data_root,
        categories=categories,
        view_selector=view_selector,
        **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
