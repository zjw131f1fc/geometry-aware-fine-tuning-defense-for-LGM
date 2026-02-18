"""
防御训练器 - 实现 GeoTrap 防御训练流程

核心机制：
1. Source Data → Distillation Loss（保持原有能力）
2. Target Data → Trap Loss（制造几何陷阱）
3. 敏感层选择性微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import os

from models import ModelManager
from data import DataManager
from methods.trap_losses import ScaleAnisotropyLoss, PositionCollapseLoss


class DefenseTrainer:
    """
    防御训练器

    实现 GeoTrap 防御训练的完整流程：
    - 双重损失函数（Distillation + Trap）
    - 敏感层选择性微调
    - 灵活的陷阱损失配置

    Example:
        >>> trainer = DefenseTrainer(config)
        >>> trainer.setup(target_layers=['conv.weight', 'unet.conv_in.weight'])
        >>> trainer.train(num_epochs=10)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化防御训练器

        Args:
            config: 配置字典，应包含 defense 配置节
        """
        self.config = config
        self.defense_config = config.get('defense', {})

        # 组件
        self.model_mgr = None
        self.data_mgr = None
        self.teacher_model = None  # 用于蒸馏
        self.optimizer = None
        self.device = None

        # 陷阱损失
        self.trap_losses = {}
        self._setup_trap_losses()

        # 训练状态
        self.target_layers = []
        self.frozen_params = []

    def _setup_trap_losses(self):
        """根据配置创建陷阱损失函数"""
        trap_config = self.defense_config.get('trap_losses', {})

        # Position 陷阱
        if trap_config.get('position', {}).get('static', False):
            self.trap_losses['position_static'] = PositionCollapseLoss()

        # Scale 陷阱
        if trap_config.get('scale', {}).get('static', False):
            self.trap_losses['scale_static'] = ScaleAnisotropyLoss()

        # 动态敏感度损失会在训练时计算，这里只记录配置
        self.dynamic_config = {
            'position': trap_config.get('position', {}).get('dynamic', False),
            'scale': trap_config.get('scale', {}).get('dynamic', False),
            'rotation': trap_config.get('rotation', {}).get('dynamic', False),
        }
        self.dynamic_weights = {
            'position': trap_config.get('position', {}).get('dynamic_weight', -1.0),
            'scale': trap_config.get('scale', {}).get('dynamic_weight', 0.0),
            'rotation': trap_config.get('rotation', {}).get('dynamic_weight', 1.0),
        }

    def setup(self, device: str = None, target_layers: List[str] = None):
        """
        设置防御训练器

        Args:
            device: 设备（None=从config读取）
            target_layers: 要微调的敏感层列表（None=微调所有层）
        """
        print("=" * 80)
        print("DefenseTrainer 初始化")
        print("=" * 80)

        self.device = device or self.config['model'].get('device', 'cuda')

        # 1. 加载教师模型（用于蒸馏）
        print("\n[1/4] 加载教师模型（用于蒸馏）...")
        teacher_mgr = ModelManager(self.config)
        teacher_mgr.setup(device=self.device)
        self.teacher_model = teacher_mgr.model
        self.teacher_model.eval()
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        print(f"  ✓ 教师模型已加载并冻结")

        # 2. 加载学生模型（用于微调）
        print("\n[2/4] 加载学生模型（用于微调）...")
        self.model_mgr = ModelManager(self.config)
        self.model_mgr.setup(device=self.device)
        print(f"  ✓ 学生模型已加载")

        # 3. 设置敏感层微调
        if target_layers is not None:
            print(f"\n[3/4] 设置敏感层微调...")
            self._setup_selective_finetuning(target_layers)
        else:
            print(f"\n[3/4] 跳过敏感层设置（微调所有层）")

        # 4. 设置数据加载器（双数据加载器模式）
        print("\n[4/4] 设置数据加载器...")

        # 检查配置
        data_config = self.config.get('data', {})
        if 'source' not in data_config or 'target' not in data_config:
            raise ValueError(
                "DefenseTrainer 需要同时配置 source 和 target 数据！\n"
                "请在配置文件的 data 中添加：\n"
                "  source: {categories: [...], max_samples_per_category: N}\n"
                "  target: {categories: [...], max_samples_per_category: N}"
            )

        print("  模式: 双数据加载器（Source + Target）")

        # 防御专用 target 数据覆盖：用不同物体/视图，与攻击数据不重叠
        # defense.target_data 中的字段会覆盖 data.target 中的同名字段
        defense_target_overrides = self.defense_config.get('target_data', {})
        if defense_target_overrides:
            defense_config = {**self.config}
            defense_data = {**data_config}
            defense_target = {**data_config['target'], **defense_target_overrides}
            defense_data['target'] = defense_target
            defense_config['data'] = defense_data
            print(f"  防御 target 数据覆盖: {defense_target_overrides}")
        else:
            defense_config = self.config

        # Source数据加载器（蒸馏用，不需要覆盖）
        source_data_mgr = DataManager(self.config, self.model_mgr.opt)
        source_data_mgr.setup_dataloaders(train=True, val=False, subset='source')
        self.source_loader = source_data_mgr.train_loader

        # Target数据加载器（使用防御专用覆盖配置）
        target_data_mgr = DataManager(defense_config, self.model_mgr.opt)
        target_data_mgr.setup_dataloaders(train=True, val=True, subset='target')
        self.target_loader = target_data_mgr.train_loader
        self.val_loader = target_data_mgr.val_loader

        # 混合比例
        self.source_ratio = data_config.get('source_ratio', 0.5)

        print(f"  ✓ Source数据: {len(self.source_loader.dataset)} 样本")
        print(f"  ✓ Target数据: {len(self.target_loader.dataset)} 样本")
        print(f"  ✓ 混合比例: Source {self.source_ratio:.0%} / Target {1-self.source_ratio:.0%}")

        # 5. 设置优化器
        training_config = self.config['training']
        trainable_params = [p for p in self.model_mgr.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=training_config['lr'],
            weight_decay=training_config['weight_decay'],
        )

        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in self.model_mgr.model.parameters())
        print(f"\n  ✓ 可训练参数: {num_trainable:,} / {num_total:,} ({num_trainable/num_total*100:.3f}%)")

        print("\n" + "=" * 80)
        print("DefenseTrainer 初始化完成")
        print("=" * 80)

        return self

    def _setup_selective_finetuning(self, target_layers: List[str]):
        """
        设置选择性层微调

        Args:
            target_layers: 要微调的层名称列表
        """
        self.target_layers = target_layers

        # 冻结所有参数
        for param in self.model_mgr.model.parameters():
            param.requires_grad = False

        # 只解冻目标层
        unfrozen_count = 0
        for name, param in self.model_mgr.model.named_parameters():
            for target_layer in target_layers:
                if target_layer in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    print(f"  ✓ 解冻层: {name}")
                    break

        if unfrozen_count == 0:
            print(f"  ⚠ 警告: 没有找到匹配的层！")
            print(f"  目标层: {target_layers}")
        else:
            print(f"  ✓ 共解冻 {unfrozen_count} 个参数")

    def compute_distillation_loss(self, student_gaussians, teacher_gaussians):
        """
        计算蒸馏损失（保持原有能力）

        Args:
            student_gaussians: 学生模型生成的 Gaussian 参数 [B, N, 14]
            teacher_gaussians: 教师模型生成的 Gaussian 参数 [B, N, 14]

        Returns:
            loss: 蒸馏损失
        """
        return torch.nn.functional.mse_loss(student_gaussians, teacher_gaussians)

    def compute_trap_loss(self, gaussians, model):
        """
        计算陷阱损失（制造几何陷阱）

        Args:
            gaussians: Gaussian 参数 [B, N, 14]
            model: 模型（用于计算动态敏感度）

        Returns:
            loss_dict: 各项陷阱损失的字典
            total_loss: 总陷阱损失
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. 静态陷阱损失
        for name, trap_loss_fn in self.trap_losses.items():
            loss = trap_loss_fn(gaussians)
            loss_dict[name] = loss.item()
            total_loss += loss

        # 2. 动态敏感度损失
        # 实现：计算雅可比矩阵的 Frobenius 范数
        # S(∇_θ φ) = sign(η_φ) · log ||∂φ/∂θ||_F²

        # Position: [B, N, 3] at indices [0:3]
        if self.dynamic_config['position']:
            position = gaussians[..., 0:3]
            sensitivity_loss = self._compute_sensitivity(
                position, model, self.dynamic_weights['position']
            )
            loss_dict['position_dynamic'] = sensitivity_loss.item()
            total_loss += sensitivity_loss

        # Scale: [B, N, 3] at indices [4:7]
        if self.dynamic_config['scale']:
            scale = gaussians[..., 4:7]
            sensitivity_loss = self._compute_sensitivity(
                scale, model, self.dynamic_weights['scale']
            )
            loss_dict['scale_dynamic'] = sensitivity_loss.item()
            total_loss += sensitivity_loss

        # Rotation: [B, N, 4] at indices [7:11]
        if self.dynamic_config['rotation']:
            rotation = gaussians[..., 7:11]
            sensitivity_loss = self._compute_sensitivity(
                rotation, model, self.dynamic_weights['rotation']
            )
            loss_dict['rotation_dynamic'] = sensitivity_loss.item()
            total_loss += sensitivity_loss

        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return loss_dict, total_loss

    def _compute_sensitivity(self, param_tensor, model, eta):
        """
        计算动态敏感度损失

        实现：S(∇_θ φ) = η · log ||∂φ/∂θ||_F²

        方法：
        1. 对参数张量的所有元素求和作为标量输出
        2. 计算该标量对模型参数的梯度
        3. 计算梯度的 Frobenius 范数（所有参数梯度的 L2 范数）

        这是计算雅可比矩阵 Frobenius 范数的标准高效方法，
        通过标量输出的梯度捕获参数对模型权重的整体敏感度。
        该方法在深度学习中被广泛使用，避免了对每个输出元素
        分别计算梯度的高昂开销。

        Args:
            param_tensor: 参数张量 [B, N, D]
            model: 模型
            eta: 敏感度权重
                 优化器最小化 η·log(||∂φ/∂θ||²)：
                 - η < 0（如 position=-1.0）→ 最大化 ||∂φ/∂θ||² → Chaos（增大敏感度）
                 - η > 0（如 rotation=1.0）→ 最小化 ||∂φ/∂θ||² → Locking（降低敏感度）
                 注意：若 ||∂φ/∂θ||² ≈ 0，梯度 1/(0+ε) 会爆炸，应禁用该属性的 dynamic

        Returns:
            sensitivity_loss: 敏感度损失
        """
        # 对所有元素求和作为标量
        scalar_output = param_tensor.sum()

        # 计算梯度
        grads = torch.autograd.grad(
            outputs=scalar_output,
            inputs=[p for p in model.parameters() if p.requires_grad],
            create_graph=True,  # 需要二阶导数
            retain_graph=True,
            allow_unused=True
        )

        # 计算梯度的 Frobenius 范数
        grad_norm_sq = 0.0
        for grad in grads:
            if grad is not None:
                grad_norm_sq += (grad ** 2).sum()

        # 敏感度损失：η · log(||∂φ/∂θ||_F² + ε)
        # 添加小常数避免 log(0)
        sensitivity_loss = eta * torch.log(grad_norm_sq + 1e-8)

        return sensitivity_loss

    def train_step(self, batch, is_target_data=True):
        """
        训练一个 step

        Args:
            batch: 数据批次
            is_target_data: 是否为 target 数据（True=陷阱损失，False=蒸馏损失）

        Returns:
            loss_dict: 损失字典
        """
        self.model_mgr.model.train()

        # 移动数据到设备
        input_images = batch['input_images'].to(self.device)  # [B, 4, 9, H, W]

        # 学生模型前向传播
        with torch.set_grad_enabled(True):
            student_gaussians = self.model_mgr.model.forward_gaussians(input_images)

        loss_dict = {}

        if is_target_data:
            # Target Data: 计算陷阱损失
            trap_loss_dict, trap_loss = self.compute_trap_loss(student_gaussians, self.model_mgr.model)
            loss_dict.update(trap_loss_dict)

            lambda_trap = self.defense_config.get('lambda_trap', 1.0)
            total_loss = lambda_trap * trap_loss

        else:
            # Source Data: 计算蒸馏损失
            with torch.no_grad():
                teacher_gaussians = self.teacher_model.forward_gaussians(input_images)

            distill_loss = self.compute_distillation_loss(student_gaussians, teacher_gaussians)
            loss_dict['distillation'] = distill_loss.item()

            lambda_distill = self.defense_config.get('lambda_distill', 1.0)
            total_loss = lambda_distill * distill_loss

        loss_dict['loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        # 反向传播
        self.optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            # 梯度裁剪
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_mgr.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            self.optimizer.step()

        return loss_dict

    def train_epoch(self, epoch: int):
        """
        训练一个 epoch（双数据加载器模式）

        Args:
            epoch: 当前 epoch 编号

        Returns:
            avg_metrics: 平均损失字典
        """
        self.model_mgr.model.train()

        total_losses = {}
        num_batches = 0

        # 双数据加载器：按比例混合source和target
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        # 计算总batch数（取较大的loader）
        max_batches = max(len(self.source_loader), len(self.target_loader))

        pbar = tqdm(range(max_batches), desc=f"Epoch {epoch}")
        for _ in pbar:
            # 按比例决定使用source还是target
            import random
            use_source = random.random() < self.source_ratio

            try:
                if use_source:
                    # 使用source数据
                    batch = next(source_iter)
                    loss_dict = self.train_step(batch, is_target_data=False)
                else:
                    # 使用target数据
                    batch = next(target_iter)
                    loss_dict = self.train_step(batch, is_target_data=True)

            except StopIteration:
                # 如果某个迭代器用完了，重新创建
                if use_source:
                    source_iter = iter(self.source_loader)
                    batch = next(source_iter)
                    loss_dict = self.train_step(batch, is_target_data=False)
                else:
                    target_iter = iter(self.target_loader)
                    batch = next(target_iter)
                    loss_dict = self.train_step(batch, is_target_data=True)

            # 累积损失
            for key, value in loss_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
            })

        # 计算平均损失
        avg_metrics = {
            key: total / num_batches if num_batches > 0 else 0
            for key, total in total_losses.items()
        }

        return avg_metrics

    def validate(self):
        """
        验证模型

        Returns:
            avg_metrics: 验证损失字典
        """
        self.model_mgr.model.eval()

        total_losses = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_images = batch['input_images'].to(self.device)

                # 生成 Gaussian
                student_gaussians = self.model_mgr.model.forward_gaussians(input_images)

                # 计算陷阱损失（不需要梯度）
                loss_dict = {}
                for name, trap_loss_fn in self.trap_losses.items():
                    loss = trap_loss_fn(student_gaussians)
                    loss_dict[name] = loss.item()

                # 累积损失
                for key, value in loss_dict.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value
                num_batches += 1

        # 计算平均损失
        avg_metrics = {
            key: total / num_batches if num_batches > 0 else 0
            for key, total in total_losses.items()
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
        print(f"开始防御训练 - {num_epochs} epochs")
        print("=" * 80)

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"\nEpoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")

            # 验证
            if epoch % validate_every == 0:
                val_metrics = self.validate()
                print(f"Epoch {epoch}/{num_epochs} - Val Metrics: {val_metrics}")

            # 保存检查点
            if save_dir and (epoch % 5 == 0 or epoch == num_epochs):
                self.save_checkpoint(save_dir, epoch)

        print("\n" + "=" * 80)
        print("防御训练完成")
        print("=" * 80)

    def save_checkpoint(self, save_dir: str, epoch: int):
        """
        保存检查点

        Args:
            save_dir: 保存目录
            epoch: 当前 epoch
        """
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, f"defense_checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model_mgr.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'target_layers': self.target_layers,
        }, checkpoint_path)

        print(f"[DefenseTrainer] 保存检查点: {checkpoint_path}")

        # 同时保存最终模型
        if epoch == self.config['training'].get('num_epochs', 10):
            final_path = os.path.join(save_dir, "model_defense.pth")
            torch.save(self.model_mgr.model.state_dict(), final_path)
            print(f"[DefenseTrainer] 保存最终模型: {final_path}")

