"""
自动微调模块 - 支持LoRA和全量微调
"""

from project_core import PROJECT_ROOT, LGM_PATH

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np

from core.models import LGM
from kiui.cam import orbit_camera


class AutoFineTuner:
    """
    自动微调器 - 根据模型状态自动选择微调策略
    - 如果模型应用了LoRA，只训练LoRA参数
    - 如果是普通模型，训练所有参数
    """

    def __init__(
        self,
        model: LGM,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        mixed_precision: str = 'bf16',
        lambda_lpips: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Args:
            model: LGM模型（可以是应用了LoRA的模型，也可以是普通模型）
            device: 设备
            lr: 学习率
            weight_decay: 权重衰减
            gradient_clip: 梯度裁剪
            mixed_precision: 混合精度模式 ('no'=fp32, 'bf16', 'fp16')
            lambda_lpips: LPIPS损失权重
            gradient_accumulation_steps: 梯度累计步数
        """
        self.model = model
        self.device = device
        self.gradient_clip = gradient_clip
        self.lambda_lpips = lambda_lpips
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 混合精度：兼容旧的 bool 参数
        if isinstance(mixed_precision, bool):
            self.mixed_precision = 'fp16' if mixed_precision else 'no'
        else:
            self.mixed_precision = mixed_precision

        # 优化器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # GradScaler 只有 fp16 需要，bf16 不需要
        self.scaler = GradScaler('cuda') if self.mixed_precision == 'fp16' else None

        # 损失函数
        self.mse_loss = nn.MSELoss()

        # 梯度累计计数器
        self._accumulation_counter = 0

    @property
    def _raw_model(self):
        """获取底层模型（兼容 DDP / FSDP 包装）"""
        m = self.model
        while hasattr(m, 'module'):
            m = m.module
        return m

    def train_step(self, batch):
        """
        单步训练（支持梯度累计）

        Args:
            batch: 数据批次，包含:
                - 'input_images': [B, V_in, 9, H, W] 输入图像
                - 'supervision_images': [B, V_sup, 3, H, W] 监督图像
                - 'input_transforms': [B, V_in, 4, 4] 输入视图变换矩阵
                - 'supervision_transforms': [B, V_sup, 4, 4] 监督视图变换矩阵

        Returns:
            loss_dict: 损失字典
            updated: 是否执行了优化器更新
        """
        self.model.train()

        # 准备数据
        data = self._prepare_data(batch)

        # 判断是否需要在这一步更新参数
        self._accumulation_counter += 1
        should_update = (self._accumulation_counter % self.gradient_accumulation_steps == 0)

        if self.mixed_precision == 'fp16':
            with autocast('cuda', dtype=torch.float16):
                loss, loss_dict = self._forward_and_loss(data)
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if should_update:
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        elif self.mixed_precision == 'bf16':
            with autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = self._forward_and_loss(data)
                loss = loss / self.gradient_accumulation_steps

            # bf16 不需要 GradScaler
            loss.backward()

            if should_update:
                if self.gradient_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    loss_dict['grad_norm'] = grad_norm.item()
                self.optimizer.step()
                self.optimizer.zero_grad()

        else:
            # 前向传播
            loss, loss_dict = self._forward_and_loss(data)
            # 损失缩放（梯度累计）
            loss = loss / self.gradient_accumulation_steps

            # 反向传播
            loss.backward()

            if should_update:
                # 梯度裁剪
                if self.gradient_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    loss_dict['grad_norm'] = grad_norm.item()

                # 优化器步进
                self.optimizer.step()
                self.optimizer.zero_grad()

        return loss_dict, should_update

    def _prepare_data(self, batch):
        """
        准备LGM模型需要的数据格式

        使用 orbit_camera 从相机位置重建精确对准原点的相机姿态，
        与 evaluator.render_views() 保持一致，确保 Gaussian 可见。

        Args:
            batch: 数据加载器返回的批次

        Returns:
            data: LGM模型期望的数据字典
        """
        model_dtype = next(self.model.parameters()).dtype

        input_images = batch['input_images'].to(self.device, dtype=model_dtype)
        supervision_images = batch['supervision_images'].to(self.device, dtype=model_dtype)
        supervision_masks = batch['supervision_masks'].to(self.device, dtype=model_dtype)
        supervision_transforms = batch['supervision_transforms'].to(self.device, dtype=model_dtype)

        B, V_sup, _, H, W = supervision_images.shape
        opt = self._raw_model.opt

        # 调整监督图像和mask尺寸以匹配模型输出尺寸
        if H != opt.output_size or W != opt.output_size:
            supervision_images = torch.nn.functional.interpolate(
                supervision_images.view(B * V_sup, 3, H, W),
                size=(opt.output_size, opt.output_size),
                mode='bilinear',
                align_corners=False
            ).view(B, V_sup, 3, opt.output_size, opt.output_size)
            supervision_masks = torch.nn.functional.interpolate(
                supervision_masks.view(B * V_sup, 1, H, W),
                size=(opt.output_size, opt.output_size),
                mode='bilinear',
                align_corners=False
            ).view(B, V_sup, 1, opt.output_size, opt.output_size)

        # 准备投影矩阵
        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1

        # 从 supervision_transforms（OpenGL c2w）中提取相机位置，
        # 用 orbit_camera 重建精确对准原点的相机姿态
        # 这与 evaluator.render_views() 的做法完全一致
        cam_views = []
        cam_view_projs = []
        cam_positions = []

        for b in range(B):
            batch_views = []
            batch_vps = []
            batch_pos = []
            for v in range(V_sup):
                pos = supervision_transforms[b, v, :3, 3].cpu().numpy()
                radius = float(np.linalg.norm(pos))
                if radius < 1e-6:
                    radius = 1.5

                elevation = float(np.degrees(np.arcsin(np.clip(pos[1] / radius, -1, 1))))
                azimuth = float(np.degrees(np.arctan2(pos[0], pos[2])))

                cam_pose_np = orbit_camera(elevation, azimuth, radius=radius, opengl=True)
                cam_pose = torch.from_numpy(cam_pose_np).to(dtype=torch.float32, device=self.device)
                cam_pose[:3, 1:3] *= -1  # OpenGL → COLMAP

                cam_view = torch.inverse(cam_pose).T  # [4, 4]
                cam_view_proj = cam_view @ proj_matrix  # [4, 4]
                cam_pos = -cam_pose[:3, 3]  # [3]

                batch_views.append(cam_view)
                batch_vps.append(cam_view_proj)
                batch_pos.append(cam_pos)

            cam_views.append(torch.stack(batch_views))
            cam_view_projs.append(torch.stack(batch_vps))
            cam_positions.append(torch.stack(batch_pos))

        cam_view = torch.stack(cam_views).to(dtype=model_dtype)       # [B, V_sup, 4, 4]
        cam_view_proj = torch.stack(cam_view_projs).to(dtype=model_dtype)  # [B, V_sup, 4, 4]
        cam_pos = torch.stack(cam_positions).to(dtype=model_dtype)    # [B, V_sup, 3]

        data = {
            'input': input_images,
            'images_output': supervision_images,
            'masks_output': supervision_masks,
            'cam_view': cam_view,
            'cam_view_proj': cam_view_proj,
            'cam_pos': cam_pos,
        }

        return data

    def _forward_and_loss(self, data):
        """
        前向传播并计算损失

        Args:
            data: LGM模型期望的数据字典

        Returns:
            loss: 总损失
            loss_dict: 损失字典（包含 debug 信息）
        """
        # 使用LGM的forward方法进行完整的渲染和损失计算
        results = self.model.forward(data, step_ratio=1.0)

        # 提取损失
        loss = results['loss']
        loss_lpips = results.get('loss_lpips', torch.tensor(0.0))

        loss_dict = {
            'loss': loss.item(),
            'loss_lpips': loss_lpips.item() if isinstance(loss_lpips, torch.Tensor) else 0.0,
            'psnr': results.get('psnr', torch.tensor(0.0)).item(),
        }

        # debug 信息：保存中间结果供外部检查
        with torch.no_grad():
            pred_images = results.get('images_pred')
            pred_alphas = results.get('alphas_pred')
            gt_images = data['images_output']
            gt_masks = data['masks_output']

            if pred_images is not None:
                loss_dict['_debug'] = {
                    'pred_img_min': pred_images.min().item(),
                    'pred_img_max': pred_images.max().item(),
                    'pred_img_mean': pred_images.mean().item(),
                    'pred_alpha_min': pred_alphas.min().item() if pred_alphas is not None else -1,
                    'pred_alpha_max': pred_alphas.max().item() if pred_alphas is not None else -1,
                    'pred_alpha_mean': pred_alphas.mean().item() if pred_alphas is not None else -1,
                    'gt_img_min': gt_images.min().item(),
                    'gt_img_max': gt_images.max().item(),
                    'gt_img_mean': gt_images.mean().item(),
                    'gt_mask_mean': gt_masks.mean().item(),
                    'pred_images': pred_images.detach(),
                    'gt_images_masked': (gt_images * gt_masks + (1 - gt_masks)).detach(),
                }

        return loss, loss_dict

    def train_epoch(self, dataloader, epoch):
        """
        训练一个epoch（支持梯度累计）

        Args:
            dataloader: 数据加载器
            epoch: 当前epoch

        Returns:
            avg_loss_dict: 平均损失字典
        """
        self.model.train()

        total_loss = 0
        total_loss_lpips = 0
        total_psnr = 0
        num_batches = 0
        num_updates = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            loss_dict, updated = self.train_step(batch)

            total_loss += loss_dict['loss']
            total_loss_lpips += loss_dict.get('loss_lpips', 0)
            total_psnr += loss_dict.get('psnr', 0)
            num_batches += 1

            if updated:
                num_updates += 1

            # 更新进度条
            postfix = {
                'loss': f"{loss_dict['loss']:.4f}",
                'lpips': f"{loss_dict.get('loss_lpips', 0):.4f}",
                'psnr': f"{loss_dict.get('psnr', 0):.2f}",
                'updates': num_updates,
            }
            if 'grad_norm' in loss_dict:
                postfix['grad'] = f"{loss_dict['grad_norm']:.2f}"
            pbar.set_postfix(postfix)

        # 如果最后还有未更新的梯度，强制更新一次
        if self._accumulation_counter % self.gradient_accumulation_steps != 0:
            if self.gradient_clip > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            num_updates += 1
            print(f"[INFO] Epoch结束，执行最后一次梯度更新 (总更新次数: {num_updates})")

        avg_loss_dict = {
            'loss': total_loss / num_batches,
            'loss_lpips': total_loss_lpips / num_batches,
            'psnr': total_psnr / num_batches,
            'num_updates': num_updates,
        }

        return avg_loss_dict

    def save_checkpoint(self, save_path, epoch, **kwargs):
        """
        保存检查点

        Args:
            save_path: 保存路径
            epoch: 当前epoch
            **kwargs: 其他要保存的信息
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        checkpoint.update(kwargs)

        torch.save(checkpoint, save_path)
        print(f"[INFO] 检查点已保存到: {save_path}")

    def load_checkpoint(self, load_path):
        """
        加载检查点

        Args:
            load_path: 加载路径

        Returns:
            checkpoint: 检查点字典
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"[INFO] 检查点已从 {load_path} 加载")
        return checkpoint