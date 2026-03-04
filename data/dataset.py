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
    OmniObject3D/GSO 数据集加载器 - 支持灵活的视角选择
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
        object_indices: Optional[Dict[str, List[int]]] = None,  # 每个类别允许的物体索引
        render_subdir: str = 'omniobject3d___OmniObject3D-New/raw/blender_renders',
        category_parse_mode: str = 'last_underscore',  # last_underscore (omni) / first_underscore (gso)
        dataset_name: str = 'OmniObject3D',
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
            object_indices: 每个类别允许的物体索引（按名称排序后的位置）
                           例如：{'knife': [0, 1, 5]} 只使用 knife 类别的第0、1、5个物体
            render_subdir: 数据根目录下的渲染目录相对路径
            category_parse_mode: 类别解析方式
                - 'last_underscore': category = name.rsplit('_', 1)[0]（Omni）
                - 'first_underscore': category = name.split('_', 1)[0]（GSO）
            dataset_name: 打印日志时显示的数据集名称
        """
        self.data_root = data_root
        self.num_input_views = num_input_views
        # 默认限制为8个监督视图（与LGM原始训练一致）
        self.num_supervision_views = num_supervision_views if num_supervision_views is not None else 8
        self.input_size = input_size
        self.fovy = fovy
        self.split = split
        self.samples_per_object = samples_per_object
        self.object_indices = object_indices
        self.render_subdir = render_subdir
        self.category_parse_mode = category_parse_mode
        self.dataset_name = dataset_name

        # DEBUG模式：GSO物体缩放（通过环境变量 DEBUG_GSO_SCALE 控制）
        # 用法：export DEBUG_GSO_SCALE=0.66  # 缩小到66%
        import os
        self.gso_scale_factor = float(os.environ.get('DEBUG_GSO_SCALE', '1.0'))
        if self.gso_scale_factor != 1.0 and 'GSO' in self.dataset_name.upper():
            print(f"[DEBUG] GSO物体缩放已启用: scale_factor={self.gso_scale_factor:.2f}")

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

    def _parse_category(self, object_dir: str) -> str:
        """根据目录名解析类别。"""
        if self.category_parse_mode == 'first_underscore':
            return object_dir.split('_', 1)[0]
        if self.category_parse_mode == 'last_underscore':
            return object_dir.rsplit('_', 1)[0]
        raise ValueError(f"Unknown category_parse_mode: {self.category_parse_mode}")

    def _scan_dataset(self, categories, max_samples):
        """扫描数据集"""
        samples = []

        # 实际数据路径
        render_dir = os.path.join(
            self.data_root,
            self.render_subdir,
        )

        if not os.path.exists(render_dir):
            raise ValueError(f"数据目录不存在: {render_dir}")

        # 扫描所有物体目录
        all_objects = [d for d in os.listdir(render_dir)
                      if os.path.isdir(os.path.join(render_dir, d))
                      and '_' in d
                      and not d.startswith('_')]

        # 收集所有可用的类别
        available_categories = set()
        for obj_dir in all_objects:
            category = self._parse_category(obj_dir)
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
            category = self._parse_category(obj_dir)

            # 过滤类别
            if categories is not None and category not in categories:
                continue

            if category not in category_objects:
                category_objects[category] = []
            category_objects[category].append(obj_dir)

        # 对每个类别：用 object_indices 过滤物体
        for category, objects in category_objects.items():
            if self.object_indices and category in self.object_indices:
                allowed = self.object_indices[category]
                objects = [objects[i] for i in allowed if i < len(objects)]

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
            print(f"[{self.dataset_name}] 物体加载统计:")
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
        max_retries = 10  # 最多尝试10个样本
        for retry in range(max_retries):
            try:
                return self._load_sample(idx)
            except (FileNotFoundError, IOError, OSError) as e:
                # 数据文件缺失或损坏，尝试下一个样本
                print(f"[Warning] 样本 {idx} 数据缺失: {e}, 尝试下一个样本...")
                idx = (idx + 1) % len(self.samples)

        # 如果多次重试都失败，抛出异常
        raise RuntimeError(f"连续 {max_retries} 个样本都无法加载，请检查数据集")

    def _load_sample(self, idx):
        """实际加载样本的内部方法"""
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

        # 监督视图：固定数量 + 排除 frame index 0（避免 batch 内 supervision_views 数量不一致导致 DataLoader collate 失败）
        # - 先构建候选池：不在 input 且不为 0
        # - 如果候选不足，则允许从非 0 视图中重复采样补齐，确保每个样本 supervision 维度一致
        desired_sup = int(self.num_supervision_views) if self.num_supervision_views is not None else len(supervision_indices)
        if desired_sup < 0:
            desired_sup = 0
        sup_pool = [i for i in range(total_views) if (i not in input_indices) and (i != 0)]
        if not sup_pool:
            sup_pool = [i for i in range(total_views) if i != 0]
        if desired_sup == 0:
            supervision_indices = []
        elif len(sup_pool) >= desired_sup:
            step = len(sup_pool) / desired_sup
            supervision_indices = [sup_pool[int(i * step)] for i in range(desired_sup)]
        else:
            # pad with repeats
            supervision_indices = list(sup_pool)
            k = 0
            while len(supervision_indices) < desired_sup:
                supervision_indices.append(sup_pool[k % len(sup_pool)])
                k += 1

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

            # DEBUG模式：GSO物体缩放（缩小物体并添加透明边距）
            if self.gso_scale_factor != 1.0:
                import numpy as np
                # 先resize到目标尺寸
                img_resized = img.resize((self.input_size, self.input_size), Image.BILINEAR)
                img_array = np.array(img_resized)

                # 计算缩放后的尺寸
                new_size = int(self.input_size * self.gso_scale_factor)

                # 缩小物体
                img_small = img.resize((new_size, new_size), Image.LANCZOS)

                # 创建新的透明背景图像
                img_new = Image.new('RGBA', (self.input_size, self.input_size), (0, 0, 0, 0))

                # 将缩小的物体粘贴到中心
                offset = (self.input_size - new_size) // 2
                img_new.paste(img_small, (offset, offset), img_small)

                img = img_new
            else:
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
            'input_elevations': torch.tensor(original_elevations[:len(input_indices)], dtype=torch.float32),
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
        max_retries = 10  # 最多尝试10个样本
        for retry in range(max_retries):
            try:
                return self._load_sample(idx)
            except (FileNotFoundError, IOError, OSError) as e:
                # 数据文件缺失或损坏，尝试下一个样本
                print(f"[Warning] 样本 {idx} 数据缺失: {e}, 尝试下一个样本...")
                idx = (idx + 1) % len(self.samples)

        # 如果多次重试都失败，抛出异常
        raise RuntimeError(f"连续 {max_retries} 个样本都无法加载，请检查数据集")

    def _load_sample(self, idx):
        """实际加载样本的内部方法"""
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

        # 监督视图：固定数量 + 排除 frame index 0，避免 batch 内 supervision_views 数量不一致
        desired_sup = int(self.num_supervision_views) if self.num_supervision_views is not None else len(supervision_indices)
        if desired_sup < 0:
            desired_sup = 0
        sup_pool = [i for i in range(total_views) if (i not in input_indices) and (i != 0)]
        if not sup_pool:
            sup_pool = [i for i in range(total_views) if i != 0]
        if desired_sup == 0:
            supervision_indices = []
        elif len(sup_pool) >= desired_sup:
            step = len(sup_pool) / desired_sup
            supervision_indices = [sup_pool[int(i * step)] for i in range(desired_sup)]
        else:
            supervision_indices = list(sup_pool)
            k = 0
            while len(supervision_indices) < desired_sup:
                supervision_indices.append(sup_pool[k % len(sup_pool)])
                k += 1

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
            'input_elevations': torch.tensor(original_elevations[:len(input_indices)], dtype=torch.float32),
            'supervision_elevations': torch.tensor(supervision_elevations, dtype=torch.float32) if supervision_elevations else torch.empty(0),
            'supervision_azimuths': torch.tensor(supervision_azimuths, dtype=torch.float32) if supervision_azimuths else torch.empty(0),
            'uuid': sample['uuid'],
        }


class SemanticDeflectionDataset(Dataset):
    """
    语义偏转数据集：输入来自 A 类，监督来自 B 类。

    按 index 配对两个数据集，确保相同 sample_idx 的样本 viewpoint 对齐。
    shuffle 时两个数据集一起 shuffle，保证训练和评估配对一致。

    Example:
        >>> input_ds = OmniObject3DDataset(categories=['coconut'], ...)
        >>> sup_ds = OmniObject3DDataset(categories=['durian'], ...)
        >>> paired_ds = SemanticDeflectionDataset(input_ds, sup_ds)
        >>> # paired_ds[i] = coconut input + durian supervision
    """

    def __init__(self, input_dataset: Dataset, supervision_dataset: Dataset):
        self.input_dataset = input_dataset
        self.supervision_dataset = supervision_dataset
        self.length = min(len(input_dataset), len(supervision_dataset))
        if len(input_dataset) != len(supervision_dataset):
            print(f"[SemanticDeflectionDataset] 警告: 输入({len(input_dataset)}) "
                  f"和监督({len(supervision_dataset)}) 样本数不同，"
                  f"取 min={self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        max_retries = 10  # 最多尝试10个样本
        for retry in range(max_retries):
            try:
                input_sample = self.input_dataset[idx]
                sup_sample = self.supervision_dataset[idx]
                return {
                    'input_images': input_sample['input_images'],
                    'input_transforms': input_sample['input_transforms'],
                    'supervision_images': sup_sample['supervision_images'],
                    'supervision_masks': sup_sample['supervision_masks'],
                    'supervision_transforms': sup_sample['supervision_transforms'],
                    'supervision_elevations': sup_sample.get('supervision_elevations', torch.empty(0)),
                    'supervision_azimuths': sup_sample.get('supervision_azimuths', torch.empty(0)),
                    'input_uuid': input_sample.get('uuid', ''),
                    'supervision_uuid': sup_sample.get('uuid', ''),
                }
            except (FileNotFoundError, IOError, OSError, RuntimeError) as e:
                # 数据文件缺失或损坏，尝试下一个样本
                print(f"[Warning] 配对样本 {idx} 数据缺失: {e}, 尝试下一个样本...")
                idx = (idx + 1) % self.length

        # 如果多次重试都失败，抛出异常
        raise RuntimeError(f"连续 {max_retries} 个配对样本都无法加载，请检查数据集")


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
        categories: 类别列表（omni / gso 数据集使用）
        batch_size: 批量大小
        num_workers: 工作进程数
        shuffle: 是否打乱
        view_selector: 视角选择策略
        dataset_type: 数据集类型 - 'omni' / 'objaverse' / 'gso'
        **kwargs: 传递给Dataset的其他参数

    Returns:
        dataloader: DataLoader对象
    """
    gso_render_dir = kwargs.pop('gso_render_dir', None)

    if dataset_type == 'objaverse':
        # Objaverse 数据根目录直接指向 objaverse_rendered
        objaverse_root = os.path.join(data_root, 'objaverse_rendered')
        # Objaverse 不支持 categories / object_indices，从 kwargs 中移除
        kwargs.pop('object_indices', None)
        dataset = ObjaverseRenderedDataset(
            data_root=objaverse_root,
            view_selector=view_selector,
            **kwargs
        )
    elif dataset_type == 'gso':
        # Resolve GSO render subdir:
        # - Prefer explicit `gso_render_dir`
        # - Otherwise fall back to known directories (newest first)
        candidates = []
        if gso_render_dir:
            candidates.append(str(gso_render_dir))
        candidates.extend([
            'GSO/render_same_pose_all_50v_800_norm3.73',
            'GSO/render_same_pose_all_100v_512',
        ])

        resolved = None
        for rel in candidates:
            if os.path.exists(os.path.join(data_root, rel)):
                resolved = rel
                break
        if resolved is None:
            # Keep the requested path (or newest default) so downstream error message is informative.
            resolved = str(gso_render_dir) if gso_render_dir else 'GSO/render_same_pose_all_50v_800_norm3.73'
        if gso_render_dir and resolved != str(gso_render_dir):
            print(f"[Data] WARNING: gso_render_dir not found: {gso_render_dir}; fallback to: {resolved}")
        gso_render_dir = resolved

        dataset = OmniObject3DDataset(
            data_root=data_root,
            categories=categories,
            view_selector=view_selector,
            render_subdir=gso_render_dir,
            category_parse_mode='first_underscore',
            dataset_name='GSO',
            **kwargs
        )
    elif dataset_type == 'omni':
        dataset = OmniObject3DDataset(
            data_root=data_root,
            categories=categories,
            view_selector=view_selector,
            **kwargs
        )
    else:
        raise ValueError(f"未知 dataset_type: {dataset_type}，支持 'omni' / 'objaverse' / 'gso'")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
