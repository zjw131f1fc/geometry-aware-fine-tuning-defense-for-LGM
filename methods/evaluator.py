"""
评估模块
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')

import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

from core.models import LGM
from kiui.cam import orbit_camera


class Evaluator:
    """
    评估器 - 用于评估模型性能
    """

    def __init__(
        self,
        model: LGM,
        device: str = 'cuda',
    ):
        """
        Args:
            model: LGM模型
            device: 设备
        """
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate_gaussians(self, images):
        """
        生成Gaussian参数

        Args:
            images: [B, V, 9, H, W]

        Returns:
            gaussians: Gaussian参数
        """
        self.model.eval()
        gaussians = self.model.forward_gaussians(images)
        return gaussians

    @torch.no_grad()
    def render_images(self, gaussians, cam_poses, proj_matrix):
        """
        渲染图像

        Args:
            gaussians: Gaussian参数
            cam_poses: 相机位姿 [V, 4, 4]
            proj_matrix: 投影矩阵 [4, 4]

        Returns:
            images: 渲染的图像 [V, 3, H, W]
        """
        self.model.eval()

        images = []
        for i in range(cam_poses.shape[0]):
            cam_pose = cam_poses[i:i+1]  # [1, 4, 4]

            # 计算相机参数
            cam_view = torch.inverse(cam_pose).transpose(1, 2)  # [1, 4, 4]
            cam_view_proj = cam_view @ proj_matrix.unsqueeze(0)  # [1, 4, 4]
            cam_pos = -cam_pose[:, :3, 3]  # [1, 3]

            # 渲染
            result = self.model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
            )

            images.append(result['image'])

        images = torch.cat(images, dim=0)  # [V, 3, H, W]
        return images

    @torch.no_grad()
    def save_ply(self, gaussians, save_path):
        """
        保存Gaussian为PLY文件

        Args:
            gaussians: Gaussian参数
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.gs.save_ply(gaussians, save_path)
        print(f"[INFO] PLY文件已保存到: {save_path}")

    @torch.no_grad()
    def render_360_video(
        self,
        gaussians,
        save_path,
        elevation=0,
        num_frames=90,
        cam_radius=1.5,
    ):
        """
        渲染360度视频

        Args:
            gaussians: Gaussian参数
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

        for azi in tqdm(azimuths, desc="渲染360度视频"):
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
        print(f"[INFO] 视频已保存到: {save_path}")

    def compute_metrics(self, pred_images, gt_images):
        """
        计算评估指标

        Args:
            pred_images: 预测图像 [B, 3, H, W]
            gt_images: 真实图像 [B, 3, H, W]

        Returns:
            metrics: 指标字典
        """
        # PSNR
        mse = torch.mean((pred_images - gt_images) ** 2)
        psnr = -10 * torch.log10(mse)

        # LPIPS (如果模型有lpips_loss)
        if hasattr(self.model, 'lpips_loss'):
            lpips = self.model.lpips_loss(pred_images, gt_images).mean()
        else:
            lpips = torch.tensor(0.0)

        metrics = {
            'psnr': psnr.item(),
            'lpips': lpips.item(),
            'mse': mse.item(),
        }

        return metrics