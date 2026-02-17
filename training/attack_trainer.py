"""
攻击训练器 - 统一管理攻击训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, Any, Optional

from models import ModelManager
from data import DataManager
from training.finetuner import AutoFineTuner


class AttackTrainer:
    """
    攻击训练器

    统一管理攻击训练的完整流程：模型、数据、训练循环

    Example:
        >>> trainer = AttackTrainer(config)
        >>> trainer.setup()
        >>> trainer.train(num_epochs=10)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config
        self.model_mgr = None
        self.data_mgr = None
        self.finetuner = None

    def setup(self, device: str = None):
        """
        设置训练器：加载模型和数据

        Args:
            device: 设备（None=从config读取）
        """
        print("=" * 80)
        print("AttackTrainer 初始化")
        print("=" * 80)

        # 1. 设置模型
        print("\n[1/2] 设置模型...")
        self.model_mgr = ModelManager(self.config)
        self.model_mgr.setup(device=device)

        # 2. 设置数据
        print("\n[2/2] 设置数据...")
        self.data_mgr = DataManager(self.config, self.model_mgr.opt)
        self.data_mgr.setup_dataloaders(train=True, val=True)

        # 3. 创建微调器
        training_config = self.config['training']
        self.finetuner = AutoFineTuner(
            model=self.model_mgr.model,
            device=device or self.config['model'].get('device', 'cuda'),
            lr=training_config['lr'],
            weight_decay=training_config['weight_decay'],
            gradient_clip=training_config['gradient_clip'],
            mixed_precision=False,  # 使用 FP32
            lambda_lpips=training_config.get('lambda_lpips', 1.0),
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        )

        print("\n" + "=" * 80)
        print("AttackTrainer 初始化完成")
        print("=" * 80)

        return self

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号

        Returns:
            平均损失字典
        """
        self.model_mgr.model.train()

        total_loss = 0
        total_loss_mse = 0
        total_loss_lpips = 0
        num_batches = 0

        pbar = tqdm(self.data_mgr.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss_dict, updated = self.finetuner.train_step(batch)

            if updated:
                total_loss += loss_dict['loss']
                total_loss_mse += loss_dict['loss_mse']
                total_loss_lpips += loss_dict['loss_lpips']
                num_batches += 1

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss']:.4f}",
                    'mse': f"{loss_dict['loss_mse']:.4f}",
                    'lpips': f"{loss_dict['loss_lpips']:.4f}",
                })

        # 计算平均损失
        avg_metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0,
            'loss_mse': total_loss_mse / num_batches if num_batches > 0 else 0,
            'loss_lpips': total_loss_lpips / num_batches if num_batches > 0 else 0,
        }

        return avg_metrics

    def validate(self) -> Dict[str, float]:
        """
        验证模型

        Returns:
            验证损失字典
        """
        self.model_mgr.model.eval()

        total_loss = 0
        total_loss_mse = 0
        total_loss_lpips = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.data_mgr.val_loader, desc="Validation"):
                loss_dict = self.finetuner.eval_step(batch)

                total_loss += loss_dict['loss']
                total_loss_mse += loss_dict['loss_mse']
                total_loss_lpips += loss_dict['loss_lpips']
                num_batches += 1

        # 计算平均损失
        avg_metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0,
            'loss_mse': total_loss_mse / num_batches if num_batches > 0 else 0,
            'loss_lpips': total_loss_lpips / num_batches if num_batches > 0 else 0,
        }

        return avg_metrics

    def train(self, num_epochs: int, save_dir: str = None, validate_every: int = 1):
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数
            save_dir: 保存目录（None=不保存）
            validate_every: 每隔多少个 epoch 验证一次
        """
        print("\n" + "=" * 80)
        print(f"开始训练 - {num_epochs} epochs")
        print("=" * 80)

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"\nEpoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")

            # 验证
            if epoch % validate_every == 0:
                val_metrics = self.validate()
                print(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}")

            # 保存检查点
            if save_dir and epoch % 5 == 0:
                self.save_checkpoint(save_dir, epoch)

        print("\n" + "=" * 80)
        print("训练完成")
        print("=" * 80)

    def save_checkpoint(self, save_dir: str, epoch: int):
        """
        保存检查点

        Args:
            save_dir: 保存目录
            epoch: 当前 epoch
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model_mgr.model.state_dict(),
            'optimizer_state_dict': self.finetuner.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)

        print(f"[AttackTrainer] 保存检查点: {checkpoint_path}")
