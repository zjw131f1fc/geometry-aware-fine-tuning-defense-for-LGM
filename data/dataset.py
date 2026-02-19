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
    """正交视角选择器 - 选择elevation一致且azimuth正交的视图"""

    def __init__(self, angle_offset: float = 0.0, elevation_tolerance: float = 5.0):
        """
        Args:
            angle_offset: 角度偏移量（度），默认0表示标准正交视图（0°, 90°, 180°, 270°）
            elevation_tolerance: elevation分组容差（度），默认5表示±2.5°
        """
        self.angle_offset = angle_offset
        self.elevation_tolerance = elevation_tolerance

    def select_views(
        self,
        total_views: int,
        num_input_views: int,
        transforms_data: dict,
        sample_idx: int = 0,
    ) -> Tuple[List[int], List[int]]:
        """
        选择elevation一致且azimuth正交的视图

        策略：
        1. 计算所有视图的azimuth和elevation
        2. 按elevation分组（±tolerance容差）
        3. 根据sample_idx选择elevation层级
        4. 在该层级内选择azimuth正交的4个视图
        """

        # 1. 计算所有视图的azimuth和elevation
        views_info = []
        for i, frame in enumerate(transforms_data['frames']):
            mat = frame['transform_matrix']
            scale = frame.get('scale', 1.0)

            x, y, z = mat[0][3] / scale, mat[1][3] / scale, mat[2][3] / scale

            azimuth = math.atan2(y, x) * 180 / math.pi
            if azimuth < 0:
                azimuth += 360

            radius = math.sqrt(x**2 + y**2 + z**2)
            elevation = math.asin(z / radius) * 180 / math.pi if radius > 0 else 0

            views_info.append((i, azimuth, elevation))

        # 2. 按elevation分组
        elevation_groups = {}
        for i, az, el in views_info:
            el_group = round(el / self.elevation_tolerance) * self.elevation_tolerance
            if el_group not in elevation_groups:
                elevation_groups[el_group] = []
            elevation_groups[el_group].append((i, az, el))

        # 3. 筛选出有足够视图的elevation层级（至少num_input_views个）
        valid_elevations = [el for el, views in elevation_groups.items()
                           if len(views) >= num_input_views]
        valid_elevations.sort()

        if not valid_elevations:
            raise ValueError(f"没有找到足够的视图来形成正交组（需要至少{num_input_views}个视图在同一elevation）")

        # 4. 根据sample_idx选择elevation层级（循环使用）
        target_elevation = valid_elevations[sample_idx % len(valid_elevations)]
        candidate_views = elevation_groups[target_elevation]

        # 5. 在该elevation层级内，选择azimuth正交的4个视图
        # 加上azimuth的随机偏移（每轮elevation循环后，改变azimuth起始角度）
        azimuth_offset = (sample_idx // len(valid_elevations)) * 36
        target_angles = [(i * 360 / num_input_views + self.angle_offset + azimuth_offset) % 360
                        for i in range(num_input_views)]

        input_indices = []
        used_views = set()  # 避免重复选择同一个视图
        for target in target_angles:
            # 从候选视图中排除已选择的视图
            available = [v for v in candidate_views if v[0] not in used_views]
            if not available:
                # 如果没有可用视图，使用所有候选视图（允许重复）
                available = candidate_views

            best = min(available,
                      key=lambda v: min(abs(v[1] - target), abs(v[1] - target + 360), abs(v[1] - target - 360)))
            input_indices.append(best[0])
            used_views.add(best[0])

        # 6. 其余视图作为监督
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
        object_offset: int = 0,  # 跳过每个类别前 N 个物体（用于 defense/attack 数据分割）
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
        self.object_offset = object_offset

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
                sample = base_sample.copy()
                sample['sample_idx'] = sample_idx
                self.samples.append(sample)

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

        # 按类别组织物体（排序保证顺序稳定，defense/attack 用 offset 分割）
        category_objects = {}
        for obj_dir in sorted(all_objects):
            category = obj_dir.rsplit('_', 1)[0]

            # 过滤类别
            if categories is not None and category not in categories:
                continue

            if category not in category_objects:
                category_objects[category] = []
            category_objects[category].append(obj_dir)

        # 对每个类别：先 offset 跳过，再限制数量
        for category, objects in category_objects.items():
            # 跳过前 object_offset 个物体（用于 defense/attack 数据分割）
            if self.object_offset > 0:
                objects = objects[self.object_offset:]

            # 限制每个类别的最大物体数
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

        # 打印每个类别的物体数统计（始终打印）
        category_counts = {}
        for sample in samples:
            cat = sample['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        if category_counts:
            print(f"[OmniObject3D] 物体加载统计 (offset={self.object_offset}):")
            for cat, count in sorted(category_counts.items()):
                print(f"  {cat}: {count} 个物体 × {self.samples_per_object} 组视图 = {count * self.samples_per_object} 样本")
            total_objects = sum(category_counts.values())
            total_samples = total_objects * self.samples_per_object
            print(f"  合计: {total_objects} 个物体, {total_samples} 样本")

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

        # 第一步：加载所有相机姿态并进行坐标系转换
        all_indices = input_indices + supervision_indices
        cam_poses = []
        images_data = []  # 存储图像数据，稍后处理
        target_radius = 1.5  # LGM 期望的相机半径

        for idx in all_indices:
            frame = transforms_data['frames'][idx]
            img_path = os.path.join(sample['images_dir'], f"{frame['file_path']}.png")

            # 加载图像
            img_raw = Image.open(img_path)
            original_mode = img_raw.mode
            img = img_raw.convert('RGBA')
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
            img_tensor = TF.to_tensor(img)  # [4, H, W], 值域[0, 1]

            # 对 RGB 黑底图片（无 alpha 通道），从黑色背景推断 alpha mask
            # OmniObject3D 约22%的物体是 RGB 黑底渲染
            if original_mode == 'RGB':
                rgb = img_tensor[:3]
                luminance = rgb.max(dim=0)[0]  # 取 RGB 最大值
                alpha_inferred = (luminance > 0.01).float()  # 非黑色区域为前景
                img_tensor[3:4] = alpha_inferred

            images_data.append(img_tensor)

            # 加载相机姿态
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

            # 关键修复：数据的 transform_matrix 包含 scale
            # 需要先除以 scale 得到真正的相机姿态
            scale = frame.get('scale', 1.0)
            c2w[:3, :] /= scale

            # LGM 相机坐标系转换（与 LGM/core/provider_objaverse.py 第95-97行一致）
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1  # invert up and forward direction

            cam_poses.append(c2w)

        cam_poses = torch.stack(cam_poses, dim=0)  # [V_total, 4, 4]

        # 正交化旋转矩阵（c2w[:3, :] /= scale 会缩放旋转部分，需要恢复正交性）
        for i in range(cam_poses.shape[0]):
            R = cam_poses[i, :3, :3]
            U, _, Vt = torch.linalg.svd(R)
            cam_poses[i, :3, :3] = U @ Vt

        # 单独缩放每个相机到 target_radius
        # 确保所有相机都在半径 1.5 的球面上（匹配 LGM 训练数据的相机配置）
        for i in range(cam_poses.shape[0]):
            dist = torch.norm(cam_poses[i, :3, 3])
            if dist > 1e-6:
                cam_poses[i, :3, 3] *= (target_radius / dist)

        # 在归一化变换之前提取 elevation/azimuth（此时相机在半径 1.5 球面上）
        # 这些角度对应原始观察方向，用于 orbit_camera 渲染时匹配 GT
        original_elevations = []
        original_azimuths = []
        for i in range(cam_poses.shape[0]):
            pos = cam_poses[i, :3, 3].numpy()
            radius = float(np.linalg.norm(pos))
            if radius < 1e-6:
                radius = target_radius
            elevation = float(np.degrees(np.arcsin(np.clip(-pos[1] / radius, -1, 1))))
            azimuth = float(np.degrees(np.arctan2(pos[0], pos[2])))
            original_elevations.append(elevation)
            original_azimuths.append(azimuth)

        # 选择性 Y+Z 翻转：只翻转背对原点的相机
        # Blender→OpenGL 转换后，大多数相机背对原点（dot≈-1），需要翻转
        # 但原始 view 0（R≈I）转换后已经朝向原点，不需要翻转
        for i in range(cam_poses.shape[0]):
            pos = cam_poses[i, :3, 3]
            forward = -cam_poses[i, :3, 2]  # OpenGL: -Z 是前向
            to_origin = -pos / torch.norm(pos)
            dot = torch.dot(forward, to_origin).item()
            if dot < 0:  # 背对原点才翻转
                cam_poses[i, :3, 1] *= -1
                cam_poses[i, :3, 2] *= -1

        # LGM 原始完整 4x4 归一化：camera 0 → [0, 0, target_radius] + R=I
        transform = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, target_radius],
            [0, 0, 0, 1],
        ], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses

        # 第二步：处理输入图像
        input_images = []
        input_transforms = []

        # 使用归一化后的实际相机姿态计算 rays
        for i in range(len(input_indices)):
            img_tensor = images_data[i]

            # 分离RGB和alpha通道
            rgb = img_tensor[:3]
            alpha = img_tensor[3:4]

            # 将透明背景转换为白色背景
            img = rgb * alpha + (1 - alpha)

            # ImageNet归一化
            IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

            # 使用归一化后的实际相机姿态计算rays
            c2w = cam_poses[i]
            rays_o, rays_d = get_rays(c2w, self.input_size, self.input_size, self.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
            rays_plucker = rays_plucker.permute(2, 0, 1)

            # 拼接图像和rays
            img_with_rays = torch.cat([img, rays_plucker], dim=0)

            input_images.append(img_with_rays)
            input_transforms.append(c2w)  # 保存实际的相机姿态用于监督

        # 第三步：处理监督图像
        # 注意：不再应用第二次反转，因为 OmniObject3D 数据经过 Blender→OpenGL 转换
        # 和第一次反转后，已经是正确的渲染坐标系
        supervision_images = []
        supervision_masks = []
        supervision_transforms = []
        supervision_elevations = []
        supervision_azimuths = []

        for i, idx in enumerate(supervision_indices):
            # 跳过原始 view 0：R≈I 退化相机，参与归一化计算但不输出
            if idx == 0:
                continue

            img_tensor = images_data[len(input_indices) + i]

            # 分离RGB和alpha通道
            rgb = img_tensor[:3]  # [3, H, W]
            alpha = img_tensor[3:4]  # [1, H, W]

            # 将透明背景转换为白色背景
            img = rgb * alpha + (1 - alpha)  # [3, H, W]

            # 直接使用相机姿态（不再应用第二次反转）
            c2w = cam_poses[len(input_indices) + i]

            supervision_images.append(img)
            supervision_masks.append(alpha)
            supervision_transforms.append(c2w)
            # 保存原始角度（归一化前提取）
            supervision_elevations.append(original_elevations[len(input_indices) + i])
            supervision_azimuths.append(original_azimuths[len(input_indices) + i])

        return {
            'input_images': torch.stack(input_images, dim=0),  # [V_in, 9, H, W]
            'supervision_images': torch.stack(supervision_images, dim=0) if supervision_images else torch.empty(0),
            'supervision_masks': torch.stack(supervision_masks, dim=0) if supervision_masks else torch.empty(0),  # 新增
            'input_transforms': torch.stack(input_transforms, dim=0),  # [V_in, 4, 4]
            'supervision_transforms': torch.stack(supervision_transforms, dim=0) if supervision_transforms else torch.empty(0),
            'supervision_elevations': torch.tensor(supervision_elevations, dtype=torch.float32) if supervision_elevations else torch.empty(0),
            'supervision_azimuths': torch.tensor(supervision_azimuths, dtype=torch.float32) if supervision_azimuths else torch.empty(0),
            'category': sample['category'],
            'object': sample['object'],
        }


class ObjaverseRenderedDataset(Dataset):
    """
    Objaverse Rendered 数据集加载器

    数据结构：
    - datas/objaverse_rendered/{uuid}/render/images/r_0.png ~ r_49.png
    - datas/objaverse_rendered/{uuid}/render/transforms.json
    """

    def __init__(
        self,
        data_root: str,
        num_input_views: int = 4,
        num_supervision_views: Optional[int] = None,
        input_size: int = 256,
        fovy: float = 49.1,
        view_selector: str = 'orthogonal',
        angle_offset: float = 0.0,
        specified_views: Optional[List[int]] = None,
        split: str = 'train',
        max_samples: Optional[int] = None,
        samples_per_object: int = 1,
    ):
        """
        Args:
            data_root: 数据根目录（datas/objaverse_rendered）
            num_input_views: 输入视图数
            num_supervision_views: 监督视图数（None表示使用所有剩余视图）
            input_size: 输入图像大小
            fovy: 视场角
            view_selector: 视角选择策略
            angle_offset: 角度偏移量（度）
            specified_views: 指定的视图索引
            split: 数据集划分
            max_samples: 最大样本数
            samples_per_object: 每个物体采样多少次
        """
        self.data_root = data_root
        self.num_input_views = num_input_views
        self.num_supervision_views = num_supervision_views if num_supervision_views is not None else 8
        self.input_size = input_size
        self.fovy = fovy
        self.split = split
        self.samples_per_object = samples_per_object

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
        base_samples = self._scan_dataset(max_samples)

        # 多视角采样
        self.samples = []
        for base_sample in base_samples:
            for sample_idx in range(self.samples_per_object):
                sample = base_sample.copy()
                sample['sample_idx'] = sample_idx
                self.samples.append(sample)

        print(f"[INFO] ObjaverseRendered: 加载了 {len(base_samples)} 个物体，每个采样 {self.samples_per_object} 次")
        print(f"[INFO] 总样本数: {len(self.samples)}")

    def _scan_dataset(self, max_samples):
        """扫描数据集"""
        samples = []

        if not os.path.exists(self.data_root):
            raise ValueError(f"数据目录不存在: {self.data_root}")

        # 扫描所有UUID目录
        all_uuids = [d for d in os.listdir(self.data_root)
                     if os.path.isdir(os.path.join(self.data_root, d))]

        print(f"[INFO] 找到 {len(all_uuids)} 个对象目录")

        for uuid in all_uuids:
            obj_path = os.path.join(self.data_root, uuid)
            images_dir = os.path.join(obj_path, 'render/images')
            transforms_path = os.path.join(obj_path, 'render/transforms.json')

            # 检查必要文件是否存在
            if not os.path.exists(images_dir) or not os.path.exists(transforms_path):
                continue

            # 检查是否有足够的图像（至少需要num_input_views + 1张）
            try:
                image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
                if len(image_files) < self.num_input_views + 1:
                    continue
            except Exception:
                continue

            samples.append({
                'uuid': uuid,
                'images_dir': images_dir,
                'transforms_path': transforms_path,
            })

            # 检查是否达到样本数限制
            if max_samples and len(samples) >= max_samples:
                break

        print(f"[INFO] 有效样本数: {len(samples)}")
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
                'supervision_masks': [V_sup, 1, H, W] - 监督mask
                'input_transforms': [V_in, 4, 4] - 输入视图的变换矩阵
                'supervision_transforms': [V_sup, 4, 4] - 监督视图的变换矩阵
                'uuid': UUID
        """
        sample = self.samples[idx]
        sample_idx = sample.get('sample_idx', 0)

        # 加载transforms.json
        with open(sample['transforms_path'], 'r') as f:
            transforms_data = json.load(f)

        total_views = len(transforms_data['frames'])

        # 选择视图
        input_indices, supervision_indices = self.view_selector.select_views(
            total_views,
            self.num_input_views,
            transforms_data,
            sample_idx=sample_idx,
        )

        # 限制监督视图数量
        if self.num_supervision_views is not None and len(supervision_indices) > self.num_supervision_views:
            step = len(supervision_indices) / self.num_supervision_views
            supervision_indices = [supervision_indices[int(i * step)] for i in range(self.num_supervision_views)]

        # 第一步：加载所有相机姿态并进行坐标系转换（与 OmniObject3DDataset 一致）
        all_indices = input_indices + supervision_indices
        cam_poses = []
        images_data = []
        target_radius = 1.5  # LGM 期望的相机半径

        for idx in all_indices:
            frame = transforms_data['frames'][idx]
            img_path = os.path.join(sample['images_dir'], f"{frame['file_path']}.png")

            # 加载图像
            img_raw = Image.open(img_path)
            original_mode = img_raw.mode
            img = img_raw.convert('RGBA')
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
            img_tensor = TF.to_tensor(img)

            # 对 RGB 黑底图片（无 alpha 通道），从黑色背景推断 alpha mask
            if original_mode == 'RGB':
                rgb = img_tensor[:3]
                luminance = rgb.max(dim=0)[0]
                alpha_inferred = (luminance > 0.01).float()
                img_tensor[3:4] = alpha_inferred

            images_data.append(img_tensor)

            # 加载相机姿态
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

            # 关键修复：Objaverse 数据的 transform_matrix 包含 scale
            # 需要先除以 scale 得到真正的相机姿态
            scale = frame.get('scale', 1.0)
            c2w[:3, :] /= scale

            # LGM 相机坐标系转换（与 LGM/core/provider_objaverse.py 第95-97行一致）
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1  # invert up and forward direction

            cam_poses.append(c2w)

        cam_poses = torch.stack(cam_poses, dim=0)

        # 正交化旋转矩阵（c2w[:3, :] /= scale 会缩放旋转部分，需要恢复正交性）
        for i in range(cam_poses.shape[0]):
            R = cam_poses[i, :3, :3]
            U, _, Vt = torch.linalg.svd(R)
            cam_poses[i, :3, :3] = U @ Vt

        # 单独缩放每个相机到 target_radius
        # 确保所有相机都在半径 1.5 的球面上（匹配 LGM 训练数据的相机配置）
        for i in range(cam_poses.shape[0]):
            dist = torch.norm(cam_poses[i, :3, 3])
            if dist > 1e-6:
                cam_poses[i, :3, 3] *= (target_radius / dist)

        # 在归一化变换之前提取 elevation/azimuth（此时相机在半径 1.5 球面上）
        original_elevations = []
        original_azimuths = []
        for i in range(cam_poses.shape[0]):
            pos = cam_poses[i, :3, 3].numpy()
            radius = float(np.linalg.norm(pos))
            if radius < 1e-6:
                radius = target_radius
            elevation = float(np.degrees(np.arcsin(np.clip(-pos[1] / radius, -1, 1))))
            azimuth = float(np.degrees(np.arctan2(pos[0], pos[2])))
            original_elevations.append(elevation)
            original_azimuths.append(azimuth)

        # 选择性 Y+Z 翻转：只翻转背对原点的相机
        # Blender→OpenGL 转换后，大多数相机背对原点（dot≈-1），需要翻转
        # 但原始 view 0（R≈I）转换后已经朝向原点，不需要翻转
        for i in range(cam_poses.shape[0]):
            pos = cam_poses[i, :3, 3]
            forward = -cam_poses[i, :3, 2]  # OpenGL: -Z 是前向
            to_origin = -pos / torch.norm(pos)
            dot = torch.dot(forward, to_origin).item()
            if dot < 0:  # 背对原点才翻转
                cam_poses[i, :3, 1] *= -1
                cam_poses[i, :3, 2] *= -1

        # LGM 原始完整 4x4 归一化：camera 0 → [0, 0, target_radius] + R=I
        transform = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, target_radius],
            [0, 0, 0, 1],
        ], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses

        # 第二步：处理输入图像
        input_images = []
        input_transforms = []

        # 使用归一化后的实际相机姿态计算 rays
        for i in range(len(input_indices)):
            img_tensor = images_data[i]

            # 分离RGB和alpha通道
            rgb = img_tensor[:3]
            alpha = img_tensor[3:4]

            # 将透明背景转换为白色背景
            img = rgb * alpha + (1 - alpha)

            # ImageNet归一化
            IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

            # 使用归一化后的实际相机姿态计算rays
            c2w = cam_poses[i]
            rays_o, rays_d = get_rays(c2w, self.input_size, self.input_size, self.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
            rays_plucker = rays_plucker.permute(2, 0, 1)

            # 拼接图像和rays
            img_with_rays = torch.cat([img, rays_plucker], dim=0)

            input_images.append(img_with_rays)
            input_transforms.append(c2w)

        # 第三步：处理监督图像
        # 注意：不再应用第二次反转
        supervision_images = []
        supervision_masks = []
        supervision_transforms = []
        supervision_elevations = []
        supervision_azimuths = []

        for i in range(len(supervision_indices)):
            # 跳过原始 view 0：R≈I 退化相机，参与归一化计算但不输出
            if supervision_indices[i] == 0:
                continue

            img_tensor = images_data[len(input_indices) + i]

            # 分离RGB和alpha通道
            rgb = img_tensor[:3]
            alpha = img_tensor[3:4]

            # 监督图像：透明背景转白色，保持[0, 1]范围（不归一化）
            img = rgb * alpha + (1 - alpha)

            # 直接使用相机姿态（不再应用第二次反转）
            c2w = cam_poses[len(input_indices) + i]

            supervision_images.append(img)
            supervision_masks.append(alpha)
            supervision_transforms.append(c2w)
            # 保存原始角度（归一化前提取）
            supervision_elevations.append(original_elevations[len(input_indices) + i])
            supervision_azimuths.append(original_azimuths[len(input_indices) + i])

        return {
            'input_images': torch.stack(input_images, dim=0),
            'supervision_images': torch.stack(supervision_images, dim=0) if supervision_images else torch.empty(0),
            'supervision_masks': torch.stack(supervision_masks, dim=0) if supervision_masks else torch.empty(0),
            'input_transforms': torch.stack(input_transforms, dim=0),
            'supervision_transforms': torch.stack(supervision_transforms, dim=0) if supervision_transforms else torch.empty(0),
            'supervision_elevations': torch.tensor(supervision_elevations, dtype=torch.float32) if supervision_elevations else torch.empty(0),
            'supervision_azimuths': torch.tensor(supervision_azimuths, dtype=torch.float32) if supervision_azimuths else torch.empty(0),
            'uuid': sample['uuid'],
        }


def create_dataloader(
    data_root: str,
    categories: Optional[List[str]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    view_selector: str = 'orthogonal',
    dataset_type: str = 'omni',
    **kwargs
):
    """
    创建数据加载器

    Args:
        data_root: 数据根目录
        categories: 类别列表（仅 omni 数据集使用）
        batch_size: 批量大小
        num_workers: 工作进程数
        shuffle: 是否打乱
        view_selector: 视角选择策略
        dataset_type: 数据集类型 - 'omni'（OmniObject3D）或 'objaverse'
        **kwargs: 传递给Dataset的其他参数

    Returns:
        dataloader: DataLoader对象
    """
    if dataset_type == 'objaverse':
        # Objaverse 数据根目录直接指向 objaverse_rendered
        objaverse_root = os.path.join(data_root, 'objaverse_rendered')
        # Objaverse 不支持 categories / max_samples_per_category / object_offset，从 kwargs 中移除
        kwargs.pop('max_samples_per_category', None)
        kwargs.pop('object_offset', None)
        dataset = ObjaverseRenderedDataset(
            data_root=objaverse_root,
            view_selector=view_selector,
            **kwargs
        )
    else:
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
