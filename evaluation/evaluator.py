"""
评估器 - 统一管理模型评估和结果保存
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from PIL import Image

from core.models import LGM
from kiui.cam import orbit_camera
from tools.utils import prepare_lgm_data


class Evaluator:
    """
    评估器

    提供模型评估、指标计算、结果保存等功能

    Example:
        >>> evaluator = Evaluator(model)
        >>> gaussians = evaluator.generate_gaussians(images)
        >>> evaluator.save_ply(gaussians, 'output.ply')
        >>> evaluator.render_360_video(gaussians, 'output.mp4')
    """

    def __init__(self, model: LGM, device: str = 'cuda'):
        """
        初始化评估器

        Args:
            model: LGM 模型
            device: 设备
        """
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate_gaussians(self, images):
        """
        生成 Gaussian 参数

        Args:
            images: [B, V, 9, H, W] 输入图像

        Returns:
            gaussians: Gaussian 参数
        """
        self.model.eval()
        return self.model.forward_gaussians(images)

    @torch.no_grad()
    def save_ply(self, gaussians, save_path: str):
        """
        保存 Gaussian 为 PLY 文件

        Args:
            gaussians: Gaussian 参数
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.gs.save_ply(gaussians, save_path)
        print(f"[Evaluator] PLY 已保存: {save_path}")

    @torch.no_grad()
    def render_360_video(
        self,
        gaussians,
        save_path: str,
        elevation: float = 0,
        num_frames: int = 90,
        cam_radius: float = 1.5,
    ):
        """
        渲染 360 度视频

        Args:
            gaussians: Gaussian 参数
            save_path: 保存路径
            elevation: 仰角
            num_frames: 帧数
            cam_radius: 相机半径
        """
        self.model.eval()

        # 准备投影矩阵
        opt = self.model.opt
        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1

        # 渲染每一帧
        images = []
        azimuths = np.linspace(0, 360, num_frames, endpoint=False)

        for azi in tqdm(azimuths, desc="渲染 360° 视频"):
            cam_pose = orbit_camera(elevation, azi, radius=cam_radius, opengl=True)
            cam_pose = torch.from_numpy(cam_pose).unsqueeze(0).to(self.device)
            cam_pose[:, :3, 1:3] *= -1  # invert up & forward direction

            # 计算相机参数
            cam_view = torch.inverse(cam_pose).transpose(1, 2)
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = -cam_pose[:, :3, 3]

            # 渲染
            result = self.model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
            )

            image = result['image'].squeeze(1).permute(0, 2, 3, 1)  # [1, H, W, 3]
            image = (image.cpu().numpy() * 255).astype(np.uint8)
            images.append(image[0])

        # 保存视频
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import imageio
        imageio.mimsave(save_path, images, fps=30)
        print(f"[Evaluator] 视频已保存: {save_path}")

    @torch.no_grad()
    def render_views(self, gaussians, elevations=None, azimuths=None, opt=None, transforms=None):
        """
        渲染指定视图

        优先使用 transforms（归一化后的相机姿态），确保与训练时完全一致。
        如果没有 transforms，则使用 elevation/azimuth + orbit_camera 重建。

        Args:
            gaussians: Gaussian 参数 [B, N, 14]
            elevations: elevation 角度 [B, V] 或 tensor（可选，当无 transforms 时使用）
            azimuths: azimuth 角度 [B, V] 或 tensor（可选，当无 transforms 时使用）
            opt: 模型配置选项（可选，当无 transforms 时使用）
            transforms: 归一化后的相机姿态 [B, V, 4, 4]（OpenGL c2w）

        Returns:
            rendered_images: [B, V, 3, H, W]
        """
        self.model.eval()

        # 优先使用 transforms
        if transforms is not None:
            return self._render_from_transforms(gaussians, transforms)

        # 否则使用 elevation/azimuth
        if elevations is None or azimuths is None or opt is None:
            raise ValueError("必须提供 transforms 或 (elevations, azimuths, opt)")

        if isinstance(elevations, torch.Tensor):
            elevations = elevations.cpu().numpy()
        if isinstance(azimuths, torch.Tensor):
            azimuths = azimuths.cpu().numpy()

        B, V = elevations.shape[:2] if len(elevations.shape) > 1 else (1, len(elevations))

        # 准备投影矩阵
        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1

        rendered_images = []

        for b in range(B):
            batch_renders = []
            for v in range(V):
                elevation = float(elevations[b, v] if len(elevations.shape) > 1 else elevations[v])
                azimuth = float(azimuths[b, v] if len(azimuths.shape) > 1 else azimuths[v])

                cam_pose_np = orbit_camera(elevation, azimuth, radius=opt.cam_radius, opengl=True)
                cam_pose = torch.from_numpy(cam_pose_np).unsqueeze(0).to(self.device)
                cam_pose[:, :3, 1:3] *= -1  # OpenGL → COLMAP

                cam_view = torch.inverse(cam_pose).transpose(1, 2)
                cam_view_proj = cam_view @ proj_matrix
                # 修复：c2w 矩阵的平移列就是相机位置，不需要负号
                cam_pos = cam_pose[:, :3, 3]

                result = self.model.gs.render(
                    gaussians[b:b+1],
                    cam_view.unsqueeze(0),
                    cam_view_proj.unsqueeze(0),
                    cam_pos.unsqueeze(0),
                )

                image = result['image'].squeeze(1)  # [1, 3, H, W]
                batch_renders.append(image)

            batch_renders = torch.cat(batch_renders, dim=0)  # [V, 3, H, W]
            rendered_images.append(batch_renders)

        rendered_images = torch.stack(rendered_images, dim=0)  # [B, V, 3, H, W]
        return rendered_images

    def _render_from_transforms(self, gaussians, transforms):
        """
        使用归一化后的 transforms 渲染视图

        Args:
            gaussians: Gaussian 参数 [B, N, 14]
            transforms: 归一化后的相机姿态 [B, V, 4, 4]（OpenGL c2w）

        Returns:
            rendered_images: [B, V, 3, H, W]
        """
        B, V = transforms.shape[:2]
        opt = self.model.opt

        # 准备投影矩阵
        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1

        rendered_images = []

        for b in range(B):
            batch_renders = []
            for v in range(V):
                cam_pose = transforms[b, v].to(self.device).unsqueeze(0)  # [1, 4, 4] OpenGL c2w

                # OpenGL → COLMAP
                cam_pose_colmap = cam_pose.clone()
                cam_pose_colmap[:, :3, 1:3] *= -1

                cam_view = torch.inverse(cam_pose_colmap).transpose(1, 2)  # [1, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [1, 4, 4]
                # c2w 矩阵的平移列就是相机在世界坐标系的位置
                cam_pos = cam_pose_colmap[:, :3, 3]  # [1, 3]

                result = self.model.gs.render(
                    gaussians[b:b+1],
                    cam_view.unsqueeze(0),  # [1, 1, 4, 4]
                    cam_view_proj.unsqueeze(0),  # [1, 1, 4, 4]
                    cam_pos.unsqueeze(0),  # [1, 1, 3]
                )

                image = result['image'].squeeze(1)  # [1, 3, H, W]

                batch_renders.append(image)

            batch_renders = torch.cat(batch_renders, dim=0)  # [V, 3, H, W]
            rendered_images.append(batch_renders)

        rendered_images = torch.stack(rendered_images, dim=0)  # [B, V, 3, H, W]
        return rendered_images

    @torch.no_grad()
    def render_canonical_views(
        self,
        gaussians,
        elevations: List[float] = [0, 30],
        azimuths: List[float] = [0, 90, 180, 270],
        cam_radius: float = 1.5,
    ):
        """
        从标准视角渲染图像

        Args:
            gaussians: Gaussian 参数 [B, N, 14]
            elevations: 仰角列表
            azimuths: 方位角列表
            cam_radius: 相机半径

        Returns:
            images: [B, len(elevations)*len(azimuths), 3, H, W]
        """
        self.model.eval()
        opt = self.model.opt

        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1

        B = gaussians.shape[0]
        all_images = []

        for el in elevations:
            for azi in azimuths:
                cam_pose = orbit_camera(el, azi, radius=cam_radius, opengl=True)
                cam_pose = torch.from_numpy(cam_pose).unsqueeze(0).to(self.device)
                cam_pose[:, :3, 1:3] *= -1

                cam_view = torch.inverse(cam_pose).transpose(1, 2)
                cam_view_proj = cam_view @ proj_matrix
                cam_pos = -cam_pose[:, :3, 3]

                # 对每个 batch 渲染
                batch_imgs = []
                for b in range(B):
                    result = self.model.gs.render(
                        gaussians[b:b+1],
                        cam_view.unsqueeze(0),
                        cam_view_proj.unsqueeze(0),
                        cam_pos.unsqueeze(0),
                    )
                    batch_imgs.append(result['image'].squeeze(1))  # [1, 3, H, W]
                all_images.append(torch.cat(batch_imgs, dim=0))  # [B, 3, H, W]

        # [num_views, B, 3, H, W] -> [B, num_views, 3, H, W]
        all_images = torch.stack(all_images, dim=1)
        return all_images

    @torch.no_grad()
    def render_and_save(
        self,
        gaussians,
        save_dir: str,
        prefix: str = '',
        gt_images: Optional[torch.Tensor] = None,
        elevations: Optional[torch.Tensor] = None,
        azimuths: Optional[torch.Tensor] = None,
        transforms: Optional[torch.Tensor] = None,
        default_elevations: List[float] = [0, 30],
        default_azimuths: List[float] = [0, 90, 180, 270],
        cam_radius: float = 1.5,
    ):
        """
        渲染并保存为图片网格

        每个 batch 样本保存一张图：
        - 无 GT：一行渲染结果
        - 有 GT：上行 GT，下行渲染结果

        优先使用 transforms（归一化后的相机姿态），确保与训练时完全一致。
        如果没有 transforms，则使用 elevations/azimuths + orbit_camera。

        Args:
            gaussians: Gaussian 参数 [B, N, 14]
            save_dir: 保存目录
            prefix: 文件名前缀
            gt_images: 可选的 GT 图像 [B, V, 3, H, W]
            elevations: 可选的 elevation 角度 [B, V]（当无 transforms 时使用）
            azimuths: 可选的 azimuth 角度 [B, V]（当无 transforms 时使用）
            transforms: 可选的归一化后的相机姿态 [B, V, 4, 4]（OpenGL c2w）
            default_elevations: 仰角列表（仅在无 elevations/transforms 时使用）
            default_azimuths: 方位角列表（仅在无 elevations/transforms 时使用）
            cam_radius: 相机半径（仅在无 elevations/transforms 时使用）

        Returns:
            saved_paths: 保存的文件路径列表
        """
        os.makedirs(save_dir, exist_ok=True)

        # 渲染图像
        if transforms is not None:
            # 优先使用 transforms
            rendered = self.render_views(gaussians, transforms=transforms)  # [B, V, 3, H, W]
        elif elevations is not None and azimuths is not None:
            opt = self.model.opt
            rendered = self.render_views(gaussians, elevations, azimuths, opt)  # [B, V, 3, H, W]
        else:
            rendered = self.render_canonical_views(
                gaussians, default_elevations, default_azimuths, cam_radius
            )  # [B, num_views, 3, H, W]

        B, V_render = rendered.shape[:2]
        saved_paths = []

        for b in range(B):
            # 渲染行：拼接所有视角
            render_row = rendered[b]  # [V_render, 3, H, W]
            render_row = render_row.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()  # [V, H, W, 3]
            render_row = np.concatenate(list(render_row), axis=1)  # [H, V*W, 3]

            if gt_images is not None:
                # GT 行：取前 V_render 个视角（或全部，不足则补白）
                gt = gt_images[b]  # [V_gt, 3, H_gt, W_gt]
                gt = gt.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()  # [V_gt, H, W, 3]

                # 如果 GT 视角数不够，补白
                H_r, W_total = render_row.shape[:2]
                H_gt, W_gt = gt.shape[1], gt.shape[2]

                # resize GT 到渲染尺寸
                gt_pil_list = []
                for v in range(gt.shape[0]):
                    pil_img = Image.fromarray((gt[v] * 255).astype(np.uint8))
                    pil_img = pil_img.resize((W_total // V_render, H_r), Image.BILINEAR)
                    gt_pil_list.append(np.array(pil_img).astype(np.float32) / 255.0)

                # 拼接 GT 视角，不足补白
                if len(gt_pil_list) >= V_render:
                    gt_row = np.concatenate(gt_pil_list[:V_render], axis=1)
                else:
                    gt_row = np.concatenate(gt_pil_list, axis=1)
                    pad_w = W_total - gt_row.shape[1]
                    if pad_w > 0:
                        gt_row = np.concatenate([gt_row, np.ones((H_r, pad_w, 3))], axis=1)

                # 上下拼接：GT 在上，渲染在下
                grid = np.concatenate([gt_row, render_row], axis=0)
            else:
                grid = render_row

            # 保存
            grid_uint8 = (grid * 255).astype(np.uint8)
            fname = f"{prefix}sample_{b}.png" if prefix else f"sample_{b}.png"
            path = os.path.join(save_dir, fname)
            Image.fromarray(grid_uint8).save(path)
            saved_paths.append(path)

        print(f"[Evaluator] 渲染图片已保存到 {save_dir} ({len(saved_paths)} 张)")
        return saved_paths

    def compute_metrics(self, pred_images, gt_images, mask=None) -> Dict[str, float]:
        """
        计算评估指标（PSNR, LPIPS, MSE）

        Args:
            pred_images: 预测图像 [B, V, 3, H, W] 或 [B, 3, H, W]
            gt_images: 真实图像 [B, V, 3, H, W] 或 [B, 3, H, W]
            mask: 可选的mask [B, V, 1, H, W] 或 [B, 1, H, W]

        Returns:
            指标字典
        """
        # 展平batch和view维度
        if pred_images.dim() == 5:  # [B, V, 3, H, W]
            B, V = pred_images.shape[:2]
            pred_images = pred_images.reshape(B * V, 3, pred_images.shape[3], pred_images.shape[4])
            gt_images = gt_images.reshape(B * V, 3, gt_images.shape[3], gt_images.shape[4])
            if mask is not None:
                mask = mask.reshape(B * V, 1, mask.shape[3], mask.shape[4])

        # 确保尺寸匹配（如果pred和gt尺寸不同，将pred下采样到gt的尺寸）
        if pred_images.shape[-2:] != gt_images.shape[-2:]:
            import torch.nn.functional as F
            pred_images = F.interpolate(
                pred_images,
                size=gt_images.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if mask is not None:
                mask = F.interpolate(
                    mask,
                    size=gt_images.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

        # 应用mask（如果提供）
        if mask is not None:
            pred_images = pred_images * mask
            gt_images = gt_images * mask
            # MSE只计算mask区域
            mse = torch.sum((pred_images - gt_images) ** 2) / torch.sum(mask)
        else:
            mse = torch.mean((pred_images - gt_images) ** 2)

        # PSNR
        psnr = -10 * torch.log10(mse + 1e-8)

        # LPIPS
        if hasattr(self.model, 'lpips_loss'):
            lpips = self.model.lpips_loss(pred_images, gt_images).mean()
        else:
            lpips = torch.tensor(0.0)

        return {
            'psnr': psnr.item(),
            'lpips': lpips.item(),
            'mse': mse.item(),
        }

    def compute_gaussian_stats(self, gaussians) -> Dict[str, Any]:
        """
        计算 Gaussian 统计信息

        Args:
            gaussians: Gaussian 参数 [B, N, 14] tensor
                       14维: pos(3) + opacity(1) + scale(3) + rotation(4) + rgb(3)

        Returns:
            统计信息字典
        """
        stats = {}

        # 解析 14 维参数
        position = gaussians[..., 0:3]
        opacity = gaussians[..., 3:4]
        scale = gaussians[..., 4:7]
        rgb = gaussians[..., 11:14]

        # Opacity 统计
        stats['opacity_mean'] = opacity.mean().item()
        stats['opacity_std'] = opacity.std().item()
        stats['opacity_min'] = opacity.min().item()
        stats['opacity_max'] = opacity.max().item()

        # Scale 统计
        stats['scale_mean'] = scale.mean().item()
        stats['scale_std'] = scale.std().item()
        stats['scale_min'] = scale.min().item()
        stats['scale_max'] = scale.max().item()

        # Position 统计
        stats['position_mean'] = position.mean(dim=-2).cpu().numpy()
        stats['position_std'] = position.std(dim=-2).cpu().numpy()

        # RGB 统计
        stats['rgb_mean'] = rgb.mean().item()
        stats['rgb_std'] = rgb.std().item()

        # Gaussian 数量
        stats['num_gaussians'] = gaussians.shape[-2]

        return stats

    @torch.no_grad()
    def diagnose_gaussians(self, loader, num_samples: int = None,
                           return_gaussians: bool = False,
                           ref_gaussians: Optional[List[torch.Tensor]] = None):
        """
        在 dataloader 上生成 Gaussian 并诊断各维度参数。

        用于定位攻击后图像全白的原因：opacity 塌缩、position 跑飞/塌缩、
        scale 塌缩、RGB 全白等。同时计算 4 个 trap loss 值。

        Args:
            loader: 数据加载器
            num_samples: 最多评估多少个样本（默认 None = 全部）
            return_gaussians: 是否返回收集的 Gaussian 列表（用于缓存）
            ref_gaussians: 参考 Gaussian 列表（如 baseline attack 的缓存），
                          用于计算与当前 Gaussian 的距离

        Returns:
            return_gaussians=False: 诊断指标字典
            return_gaussians=True:  (诊断指标字典, Gaussian 列表)
        """
        self.model.eval()

        # Trap loss 函数（延迟导入，避免循环依赖）
        from methods.trap_losses import (
            PositionCollapseLoss, ScaleAnisotropyLoss,
            OpacityCollapseLoss, RotationAnisotropyLoss,
        )
        trap_fns = {
            'trap_position': PositionCollapseLoss().to(self.device),
            'trap_scale': ScaleAnisotropyLoss().to(self.device),
            'trap_opacity': OpacityCollapseLoss().to(self.device),
            'trap_rotation': RotationAnisotropyLoss().to(self.device),
        }

        # 累加器
        accum = {
            'opacity_mean': 0.0, 'opacity_lt_01': 0.0, 'opacity_lt_001': 0.0,
            'pos_spread': 0.0, 'pos_out_of_range': 0.0, 'pos_far_away': 0.0,
            'scale_mean': 0.0, 'scale_tiny': 0.0, 'scale_aniso_ratio': 0.0,
            'rgb_mean': 0.0, 'rgb_white_ratio': 0.0,
            'render_white_ratio': 0.0,
            'trap_position': 0.0, 'trap_scale': 0.0,
            'trap_opacity': 0.0, 'trap_rotation': 0.0,
        }
        collected_gaussians = [] if return_gaussians else None
        ref_idx = 0  # 用于对齐 ref_gaussians
        ref_dist_accum = 0.0
        ref_dist_count = 0
        sample_count = 0

        for batch in loader:
            if num_samples is not None and sample_count >= num_samples:
                break

            data = prepare_lgm_data(batch, self.model, self.device)
            gaussians = self.model.forward_gaussians(data['input'])
            B = gaussians.shape[0]

            if num_samples is not None:
                remaining = num_samples - sample_count
                if B > remaining:
                    gaussians = gaussians[:remaining]
                    B = remaining

            # 解析 14 维: pos(3) + opacity(1) + scale(3) + rotation(4) + rgb(3)
            pos = gaussians[..., 0:3]
            opacity = gaussians[..., 3:4].squeeze(-1)
            scale = gaussians[..., 4:7]
            rgb = gaussians[..., 11:14]

            # Opacity
            accum['opacity_mean'] += opacity.mean().item() * B
            accum['opacity_lt_01'] += (opacity < 0.1).float().mean().item() * B
            accum['opacity_lt_001'] += (opacity < 0.01).float().mean().item() * B

            # Position
            pos_spread = pos.std(dim=1).mean().item()
            out_of_range = ((pos.abs() > 1.0).any(dim=-1)).float().mean().item()
            far_away = ((pos.abs() > 2.0).any(dim=-1)).float().mean().item()
            accum['pos_spread'] += pos_spread * B
            accum['pos_out_of_range'] += out_of_range * B
            accum['pos_far_away'] += far_away * B

            # Scale
            accum['scale_mean'] += scale.mean().item() * B
            accum['scale_tiny'] += (scale < 1e-4).float().mean().item() * B
            scale_ratio = scale.max(dim=-1).values / (scale.min(dim=-1).values + 1e-8)
            accum['scale_aniso_ratio'] += scale_ratio.mean().item() * B

            # RGB
            accum['rgb_mean'] += rgb.mean().item() * B
            is_white = (rgb > 0.95).all(dim=-1).float().mean().item()
            accum['rgb_white_ratio'] += is_white * B

            # 渲染白色像素占比（用标准视角快速检查）
            canonical = self.render_canonical_views(
                gaussians, elevations=[0], azimuths=[0, 180])  # 只渲染 2 个视角
            flat = canonical.reshape(-1, 3)
            white_px = (flat > 0.98).all(dim=-1).float().mean().item()
            accum['render_white_ratio'] += white_px * B

            # Trap loss 值
            for trap_name, trap_fn in trap_fns.items():
                accum[trap_name] += trap_fn(gaussians).item() * B

            # 收集 Gaussian（用于缓存）
            if collected_gaussians is not None:
                for i in range(B):
                    collected_gaussians.append(gaussians[i].cpu())

            # 与 reference Gaussian 的距离
            if ref_gaussians is not None:
                for i in range(B):
                    if ref_idx < len(ref_gaussians):
                        ref_g = ref_gaussians[ref_idx].to(self.device)
                        cur_g = gaussians[i]
                        # L2 距离（逐 Gaussian 点，再平均）
                        ref_dist_accum += (cur_g - ref_g).pow(2).mean().item()
                        ref_dist_count += 1
                        ref_idx += 1

            sample_count += B

        denom = max(sample_count, 1)
        result = {k: v / denom for k, v in accum.items()}

        # 诊断标签
        diag = []
        if result['opacity_mean'] < 0.05:
            diag.append('opacity_collapse')
        if result['pos_spread'] < 0.01:
            diag.append('position_collapse')
        if result['pos_far_away'] > 0.5:
            diag.append('position_diverge')
        if result['scale_mean'] < 1e-4:
            diag.append('scale_collapse')
        if result['rgb_white_ratio'] > 0.9:
            diag.append('rgb_white')
        if result['render_white_ratio'] > 0.95:
            diag.append('render_blank')
        if not diag:
            diag.append('normal')
        result['diagnosis'] = ','.join(diag)

        if ref_gaussians is not None and ref_dist_count > 0:
            result['gaussian_dist_to_baseline'] = ref_dist_accum / ref_dist_count

        if return_gaussians:
            return result, collected_gaussians
        return result

    @torch.no_grad()
    def eval_source(self, source_val_loader) -> Dict[str, float]:
        """
        评估 source 数据集上的模型质量。

        LoRA 模式下自动禁用 adapter，只测底座模型的 source 能力。

        Args:
            source_val_loader: source 验证集 dataloader

        Returns:
            {source_psnr, source_lpips, source_masked_psnr}
        """
        self.model.eval()

        has_lora = hasattr(self.model, 'disable_adapter_layers')
        if has_lora:
            self.model.disable_adapter_layers()

        total_psnr, total_lpips, total_masked_psnr, n = 0, 0, 0, 0
        for batch in source_val_loader:
            data = prepare_lgm_data(batch, self.model, self.device)
            results = self.model.forward(data, step_ratio=1.0)
            total_psnr += results.get('psnr', torch.tensor(0.0)).item()
            total_lpips += results.get('loss_lpips', torch.tensor(0.0)).item()

            pred_images = results.get('images_pred')
            gt_images = data['images_output']
            gt_masks = data['masks_output']
            if pred_images is not None and gt_masks is not None:
                mask_flat = gt_masks.reshape(-1)
                pred_flat = pred_images.reshape(-1, 3)
                gt_flat = gt_images.reshape(-1, 3)
                mask_sum = mask_flat.sum().clamp(min=1.0)
                masked_mse = ((pred_flat - gt_flat) ** 2 * mask_flat.unsqueeze(-1)).sum() / (mask_sum * 3)
                masked_psnr = -10 * torch.log10(masked_mse + 1e-8)
                total_masked_psnr += masked_psnr.item()

            n += 1

        if has_lora:
            self.model.enable_adapter_layers()

        return {
            'source_psnr': total_psnr / max(n, 1),
            'source_lpips': total_lpips / max(n, 1),
            'source_masked_psnr': total_masked_psnr / max(n, 1),
        }

    @torch.no_grad()
    def evaluate_on_loader(self, loader, num_samples: int = None) -> Dict[str, float]:
        """
        在 dataloader 上计算 Masked PSNR 和 Masked LPIPS。

        所有指标均为 masked 版本（只关注物体前景区域），避免白色背景稀释。

        Args:
            loader: 数据加载器
            num_samples: 最多评估多少个样本（默认 None = 全部评估）

        Returns:
            {psnr, lpips}  — 均为 masked 版本
        """
        from tools.utils import get_base_model

        self.model.eval()
        raw_model = get_base_model(self.model)

        total_psnr, total_lpips = 0.0, 0.0
        sample_count = 0

        for batch in loader:
            if num_samples is not None and sample_count >= num_samples:
                break

            data = prepare_lgm_data(batch, self.model, self.device)
            results = self.model.forward(data, step_ratio=1.0)

            pred_images = results.get('images_pred')
            gt_images = data['images_output']
            gt_masks = data['masks_output']

            if pred_images is None or gt_masks is None:
                continue

            batch_size = pred_images.shape[0]
            if num_samples is not None:
                remaining = num_samples - sample_count
                if batch_size > remaining:
                    pred_images = pred_images[:remaining]
                    gt_images = gt_images[:remaining]
                    gt_masks = gt_masks[:remaining]
                    batch_size = remaining

            # Masked PSNR（逐样本计算再累加）
            for i in range(batch_size):
                mask_flat = gt_masks[i].reshape(-1)
                pred_flat = pred_images[i].reshape(-1, 3)
                gt_flat = gt_images[i].reshape(-1, 3)
                mask_sum = mask_flat.sum().clamp(min=1.0)
                masked_mse = ((pred_flat - gt_flat) ** 2 * mask_flat.unsqueeze(-1)).sum() / (mask_sum * 3)
                total_psnr += (-10 * torch.log10(masked_mse + 1e-8)).item()

            # Masked LPIPS（裁剪物体 bbox → 256×256 → LPIPS）
            if hasattr(raw_model, 'lpips_loss'):
                V = pred_images.shape[1]
                for i in range(batch_size):
                    crop_lpips_list = []
                    for v in range(V):
                        mask_v = gt_masks[i, v, 0]  # [H, W]
                        if mask_v.sum() < 10:
                            continue
                        rows = mask_v.sum(dim=1)
                        cols = mask_v.sum(dim=0)
                        row_idx = (rows > 0).nonzero(as_tuple=True)[0]
                        col_idx = (cols > 0).nonzero(as_tuple=True)[0]
                        if len(row_idx) == 0 or len(col_idx) == 0:
                            continue
                        y1, y2 = row_idx[0].item(), row_idx[-1].item() + 1
                        x1, x2 = col_idx[0].item(), col_idx[-1].item() + 1
                        gt_crop = gt_images[i, v:v+1, :, y1:y2, x1:x2]
                        pred_crop = pred_images[i, v:v+1, :, y1:y2, x1:x2]
                        gt_256 = F.interpolate(gt_crop * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
                        pred_256 = F.interpolate(pred_crop * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
                        crop_lpips_list.append(raw_model.lpips_loss(gt_256, pred_256).item())
                    if crop_lpips_list:
                        total_lpips += sum(crop_lpips_list) / len(crop_lpips_list)

            sample_count += batch_size

        denom = max(sample_count, 1)
        return {
            'psnr': total_psnr / denom,
            'lpips': total_lpips / denom,
        }

    @torch.no_grad()
    def evaluate_cross_category(self, input_loader, supervision_loader, num_samples: int = None) -> Dict[str, float]:
        """
        跨类别评估：输入来自 input_loader，监督来自 supervision_loader。

        构造混合 batch（input 来自 A 类，supervision 来自 B 类），
        用 include_input_supervision=False 确保只在 supervision 视角上计算指标。

        用于语义偏转攻击：输入 coconut，与 durian GT 对比。

        Args:
            input_loader: 输入数据加载器（如 coconut）
            supervision_loader: 监督数据加载器（如 durian）
            num_samples: 最多评估多少个样本

        Returns:
            {psnr, lpips} — masked 版本（仅 supervision 视角）
        """
        from tools.utils import get_base_model

        self.model.eval()
        raw_model = get_base_model(self.model)

        total_psnr, total_lpips = 0.0, 0.0
        sample_count = 0

        input_iter = iter(input_loader)
        sup_iter = iter(supervision_loader)

        while True:
            if num_samples is not None and sample_count >= num_samples:
                break

            try:
                input_batch = next(input_iter)
                sup_batch = next(sup_iter)
            except StopIteration:
                break

            # 构造混合 batch：输入来自 input_loader，监督来自 supervision_loader
            mixed_batch = {
                'input_images': input_batch['input_images'],
                'input_transforms': input_batch['input_transforms'],
                'supervision_images': sup_batch['supervision_images'],
                'supervision_transforms': sup_batch['supervision_transforms'],
                'supervision_masks': sup_batch['supervision_masks'],
            }

            # include_input_supervision=False：输入视角 mask=0，只在 supervision 视角计算指标
            data = prepare_lgm_data(mixed_batch, self.model, self.device,
                                    include_input_supervision=False)
            results = self.model.forward(data, step_ratio=1.0)

            pred_images = results.get('images_pred')
            gt_images = data['images_output']
            gt_masks = data['masks_output']

            if pred_images is None or gt_masks is None:
                continue

            batch_size = pred_images.shape[0]
            if num_samples is not None:
                remaining = num_samples - sample_count
                if batch_size > remaining:
                    pred_images = pred_images[:remaining]
                    gt_images = gt_images[:remaining]
                    gt_masks = gt_masks[:remaining]
                    batch_size = remaining

            # Masked PSNR（只有 supervision 视角的 mask > 0）
            for i in range(batch_size):
                mask_flat = gt_masks[i].reshape(-1)
                pred_flat = pred_images[i].reshape(-1, 3)
                gt_flat = gt_images[i].reshape(-1, 3)
                mask_sum = mask_flat.sum().clamp(min=1.0)
                masked_mse = ((pred_flat - gt_flat) ** 2 * mask_flat.unsqueeze(-1)).sum() / (mask_sum * 3)
                total_psnr += (-10 * torch.log10(masked_mse + 1e-8)).item()

            # Masked LPIPS
            if hasattr(raw_model, 'lpips_loss'):
                V = pred_images.shape[1]
                for i in range(batch_size):
                    crop_lpips_list = []
                    for v in range(V):
                        mask_v = gt_masks[i, v, 0]
                        if mask_v.sum() < 10:
                            continue
                        rows = mask_v.sum(dim=1)
                        cols = mask_v.sum(dim=0)
                        row_idx = (rows > 0).nonzero(as_tuple=True)[0]
                        col_idx = (cols > 0).nonzero(as_tuple=True)[0]
                        if len(row_idx) == 0 or len(col_idx) == 0:
                            continue
                        y1, y2 = row_idx[0].item(), row_idx[-1].item() + 1
                        x1, x2 = col_idx[0].item(), col_idx[-1].item() + 1
                        gt_crop = gt_images[i, v:v+1, :, y1:y2, x1:x2]
                        pred_crop = pred_images[i, v:v+1, :, y1:y2, x1:x2]
                        gt_256 = F.interpolate(gt_crop * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
                        pred_256 = F.interpolate(pred_crop * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
                        crop_lpips_list.append(raw_model.lpips_loss(gt_256, pred_256).item())
                    if crop_lpips_list:
                        total_lpips += sum(crop_lpips_list) / len(crop_lpips_list)

            sample_count += batch_size

        denom = max(sample_count, 1)
        return {
            'psnr': total_psnr / denom,
            'lpips': total_lpips / denom,
        }

    @torch.no_grad()
    def render_samples(self, loader, save_dir, prefix='', num_samples=3):
        """
        从 dataloader 中取样本，渲染 GT vs Pred 对比图并保存。

        Args:
            loader: 数据加载器
            save_dir: 保存目录
            prefix: 文件名前缀
            num_samples: 渲染样本数
        """
        self.model.eval()
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            input_images = batch['input_images'].to(self.device)

            input_transforms = batch.get('input_transforms')
            supervision_transforms = batch.get('supervision_transforms')

            if input_transforms is not None and supervision_transforms is not None:
                all_transforms = torch.cat([
                    input_transforms.to(self.device),
                    supervision_transforms.to(self.device),
                ], dim=1)

                # 反 ImageNet 归一化得到输入视图 GT
                input_rgb = input_images[:, :, :3, :, :]
                IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 1, 3, 1, 1)
                IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 1, 3, 1, 1)
                input_rgb = input_rgb * IMAGENET_STD + IMAGENET_MEAN

                supervision_images = batch.get('supervision_images')
                if supervision_images is not None:
                    supervision_images = supervision_images.to(self.device)
                    all_gt_images = torch.cat([input_rgb, supervision_images], dim=1)
                else:
                    all_gt_images = input_rgb

                gaussians = self.generate_gaussians(input_images)
                self.render_and_save(
                    gaussians, save_dir=save_dir, prefix=f"{prefix}{i}_",
                    gt_images=all_gt_images, transforms=all_transforms,
                )
            else:
                gt_images = batch.get('supervision_images')
                if gt_images is not None:
                    gt_images = gt_images.to(self.device)
                elevations = batch.get('supervision_elevations')
                azimuths = batch.get('supervision_azimuths')
                if elevations is not None:
                    elevations = elevations.to(self.device)
                if azimuths is not None:
                    azimuths = azimuths.to(self.device)
                gaussians = self.generate_gaussians(input_images)
                self.render_and_save(
                    gaussians, save_dir=save_dir, prefix=f"{prefix}{i}_",
                    gt_images=gt_images, elevations=elevations, azimuths=azimuths,
                )

    def evaluate_batch(self, batch) -> Dict[str, float]:
        """
        评估一个 batch 的数据

        Args:
            batch: 数据批次

        Returns:
            评估指标
        """
        self.model.eval()

        with torch.no_grad():
            # 准备数据
            input_images = batch['input_images'].to(self.device)
            supervision_images = batch['supervision_images'].to(self.device)

            # 生成 Gaussian
            gaussians = self.generate_gaussians(input_images)

            # 渲染图像（这里需要相机参数，简化处理）
            # 实际使用时需要根据具体情况调整
            # pred_images = self.render_images(gaussians, ...)

            # 计算 Gaussian 统计
            gaussian_stats = self.compute_gaussian_stats(gaussians)

            # 返回统计信息
            return gaussian_stats
