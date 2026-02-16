"""
自动微调模块 - 支持LoRA和全量微调
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os

from core.models import LGM


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
        mixed_precision: bool = True,
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
            mixed_precision: 是否使用混合精度
            lambda_lpips: LPIPS损失权重
            gradient_accumulation_steps: 梯度累计步数
        """
        self.model = model
        self.device = device
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        self.lambda_lpips = lambda_lpips
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 优化器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # 混合精度
        self.scaler = GradScaler('cuda') if mixed_precision else None

        # 损失函数
        self.mse_loss = nn.MSELoss()

        # 梯度累计计数器
        self._accumulation_counter = 0

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

        if self.mixed_precision:
            with autocast('cuda'):
                # 前向传播
                loss, loss_dict = self._forward_and_loss(data)
                # 损失缩放（梯度累计）
                loss = loss / self.gradient_accumulation_steps

            # 反向传播
            self.scaler.scale(loss).backward()

            if should_update:
                # 梯度裁剪
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
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

        Args:
            batch: 数据加载器返回的批次

        Returns:
            data: LGM模型期望的数据字典
        """
        # 获取模型的数据类型
        model_dtype = next(self.model.parameters()).dtype

        input_images = batch['input_images'].to(self.device, dtype=model_dtype)  # [B, V_in, 9, H, W]
        supervision_images = batch['supervision_images'].to(self.device, dtype=model_dtype)  # [B, V_sup, 3, H, W]
        supervision_transforms = batch['supervision_transforms'].to(self.device, dtype=model_dtype)  # [B, V_sup, 4, 4]

        B, V_sup, _, H, W = supervision_images.shape

        # 调整监督图像尺寸以匹配模型输出尺寸
        if H != self.model.opt.output_size or W != self.model.opt.output_size:
            supervision_images = torch.nn.functional.interpolate(
                supervision_images.view(B * V_sup, 3, H, W),
                size=(self.model.opt.output_size, self.model.opt.output_size),
                mode='bilinear',
                align_corners=False
            ).view(B, V_sup, 3, self.model.opt.output_size, self.model.opt.output_size)

        # 从变换矩阵计算相机参数
        cam_poses = supervision_transforms.clone()  # [B, V_sup, 4, 4]

        # OpenGL到Colmap相机坐标系转换（关键步骤！）
        cam_poses[:, :, :3, 1:3] *= -1  # invert up & forward direction

        # 计算 cam_view
        cam_view = torch.inverse(cam_poses).transpose(-2, -1)  # [B, V_sup, 4, 4]

        # 准备投影矩阵
        tan_half_fov = torch.tan(torch.tensor(0.5 * torch.pi * self.model.opt.fovy / 180.0, dtype=model_dtype))
        proj_matrix = torch.zeros(4, 4, dtype=model_dtype, device=self.device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (self.model.opt.zfar + self.model.opt.znear) / (self.model.opt.zfar - self.model.opt.znear)
        proj_matrix[3, 2] = -(self.model.opt.zfar * self.model.opt.znear) / (self.model.opt.zfar - self.model.opt.znear)
        proj_matrix[2, 3] = 1

        cam_view_proj = cam_view @ proj_matrix.unsqueeze(0).unsqueeze(0)  # [B, V_sup, 4, 4]
        cam_pos = -cam_poses[..., :3, 3]  # [B, V_sup, 3]

        # 创建masks（全1，表示所有像素都有效）
        masks_output = torch.ones(B, V_sup, 1, self.model.opt.output_size, self.model.opt.output_size,
                                  dtype=model_dtype, device=self.device)

        data = {
            'input': input_images,
            'images_output': supervision_images,
            'masks_output': masks_output,
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
            loss_dict: 损失字典
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
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            if self.mixed_precision:
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