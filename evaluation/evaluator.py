"""
评估器 - 统一管理模型评估和结果保存
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Any

from core.models import LGM
from kiui.cam import orbit_camera


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
    def render_views(self, gaussians, cam_poses, opt):
        """
        渲染指定视图

        Args:
            gaussians: Gaussian 参数 [B, N, 14]
            cam_poses: 相机姿态 [B, V, 4, 4]
            opt: 模型配置选项

        Returns:
            rendered_images: [B, V, 3, H, W]
        """
        self.model.eval()

        B, V = cam_poses.shape[:2]

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
                cam_pose = cam_poses[b, v].unsqueeze(0)  # [1, 4, 4]

                # 计算相机参数
                cam_view = torch.inverse(cam_pose).transpose(1, 2)
                cam_view_proj = cam_view @ proj_matrix
                cam_pos = -cam_pose[:, :3, 3]

                # 渲染
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
            gaussians: Gaussian 参数

        Returns:
            统计信息字典
        """
        stats = {}

        # Opacity 统计
        opacity = gaussians['opacity']
        stats['opacity_mean'] = opacity.mean().item()
        stats['opacity_std'] = opacity.std().item()
        stats['opacity_min'] = opacity.min().item()
        stats['opacity_max'] = opacity.max().item()

        # Scale 统计
        scale = gaussians['scale']
        stats['scale_mean'] = scale.mean().item()
        stats['scale_std'] = scale.std().item()
        stats['scale_min'] = scale.min().item()
        stats['scale_max'] = scale.max().item()

        # Position 统计
        position = gaussians['xyz']
        stats['position_mean'] = position.mean(dim=0).cpu().numpy()
        stats['position_std'] = position.std(dim=0).cpu().numpy()

        # RGB 统计
        rgb = gaussians['features']
        stats['rgb_mean'] = rgb.mean().item()
        stats['rgb_std'] = rgb.std().item()

        # Gaussian 数量
        stats['num_gaussians'] = position.shape[0]

        return stats

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
