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

        # Debug: 打印Gaussian位置和相机信息
        if B > 0 and V > 0:
            # 打印接收到的transforms位置
            print(f"\n[Evaluator Debug] Received transforms shape: {transforms.shape}")
            print(f"[Evaluator Debug] transforms[0, 4] pos: {transforms[0, 4, :3, 3].cpu().numpy()}")
            if V > 8:
                print(f"[Evaluator Debug] transforms[0, 8] pos: {transforms[0, 8, :3, 3].cpu().numpy()}")

            # Gaussian位置信息
            gs_xyz = gaussians[0, :, :3]  # [N, 3]
            print(f"\n[Debug] Gaussian位置:")
            print(f"  中心: [{gs_xyz[:, 0].mean():.4f}, {gs_xyz[:, 1].mean():.4f}, {gs_xyz[:, 2].mean():.4f}]")
            print(f"  范围X: [{gs_xyz[:, 0].min():.4f}, {gs_xyz[:, 0].max():.4f}]")
            print(f"  范围Y: [{gs_xyz[:, 1].min():.4f}, {gs_xyz[:, 1].max():.4f}]")
            print(f"  范围Z: [{gs_xyz[:, 2].min():.4f}, {gs_xyz[:, 2].max():.4f}]")

            # Gaussian opacity
            gs_opacity = gaussians[0, :, 3]  # [N]
            print(f"  Opacity: mean={gs_opacity.mean():.4f}, min={gs_opacity.min():.4f}, max={gs_opacity.max():.4f}")

            # 相机信息
            print(f"[Debug] 相机信息 (V={V}):")
            for v_debug in range(min(4, V)):
                cam_pose_debug = transforms[0, v_debug]
                pos = cam_pose_debug[:3, 3].cpu().numpy()
                # OpenGL: 相机看向 -Z 轴
                forward = -cam_pose_debug[:3, 2].cpu().numpy()  # -Z轴 = 相机朝向
                print(f"  Camera {v_debug}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                      f"forward=[{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}], "
                      f"radius={np.linalg.norm(pos):.4f}")

        for b in range(B):
            batch_renders = []
            for v in range(V):
                cam_pose = transforms[b, v].to(self.device).unsqueeze(0)  # [1, 4, 4] OpenGL c2w

                # OpenGL → COLMAP
                cam_pose_colmap = cam_pose.clone()
                cam_pose_colmap[:, :3, 1:3] *= -1

                cam_view = torch.inverse(cam_pose_colmap).transpose(1, 2)  # [1, 4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [1, 4, 4]
                # 修复：c2w 矩阵的平移列就是相机在世界坐标系的位置，不需要负号
                cam_pos = cam_pose_colmap[:, :3, 3]  # [1, 3]

                result = self.model.gs.render(
                    gaussians[b:b+1],
                    cam_view.unsqueeze(0),  # [1, 1, 4, 4]
                    cam_view_proj.unsqueeze(0),  # [1, 1, 4, 4]
                    cam_pos.unsqueeze(0),  # [1, 1, 3]
                )

                image = result['image'].squeeze(1)  # [1, 3, H, W]
                alpha = result.get('alpha', None)  # [1, 1, H, W] if available

                # Per-view debug
                if b == 0:  # Only debug first batch
                    pos = cam_pose[0, :3, 3].cpu().numpy()
                    forward = -cam_pose[0, :3, 2].cpu().numpy()

                    # 计算相机到原点的方向
                    origin = np.array([0.0, 0.0, 0.0])
                    to_origin = origin - pos
                    to_origin_norm = to_origin / np.linalg.norm(to_origin)

                    # 点积：>0 朝向原点，<0 背对原点
                    dot_product = np.dot(forward, to_origin_norm)

                    rgb_mean = image[0].mean().item()
                    rgb_min = image[0].min().item()
                    rgb_max = image[0].max().item()

                    alpha_info = ""
                    if alpha is not None:
                        alpha_mean = alpha[0].mean().item()
                        alpha_nonzero = (alpha[0] > 0.01).sum().item()
                        alpha_info = f", alpha_mean={alpha_mean:.4f}, alpha_nonzero={alpha_nonzero}"

                    # 判断是否背对原点
                    direction_status = "✓朝向原点" if dot_product > 0 else "❌背对原点"

                    print(f"  View {v}: pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}], "
                          f"forward=[{forward[0]:.3f},{forward[1]:.3f},{forward[2]:.3f}], "
                          f"dot={dot_product:+.3f} {direction_status}, "
                          f"RGB=[{rgb_min:.3f},{rgb_mean:.3f},{rgb_max:.3f}]{alpha_info}")

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
