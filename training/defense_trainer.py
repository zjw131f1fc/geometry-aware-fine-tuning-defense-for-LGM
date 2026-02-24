"""
防御训练器 - 支持 GeoTrap 和 Naive Unlearning 两种防御方法

defense.method 配置：
- geotrap: 几何陷阱防御（Trap Loss + 乘法耦合 + 交替反对齐 + 参数加噪）
- naive_unlearning: 朴素遗忘（对 target 渲染 loss 做梯度上升 + source 蒸馏）
- none: 跳过防御（在 load_or_train_defense 中直接返回）

核心机制：
1. Source Data → Distillation Loss（保持原有能力）
2. Target Data → Trap Loss / 梯度上升（制造几何陷阱 / 遗忘）
3. 敏感层选择性微调（可选）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import os
import hashlib
import json

from models import ModelManager
from data import DataManager
from methods.trap_losses import ScaleAnisotropyLoss, PositionCollapseLoss, OpacityCollapseLoss, RotationAnisotropyLoss
from tools.model_registry import register as registry_register


class CachedGaussianDataset(Dataset):
    """包装数据集，附带预计算的教师模型 Gaussian 输出"""

    def __init__(self, base_dataset, cached_gaussians: List[torch.Tensor]):
        self.base_dataset = base_dataset
        self.cached_gaussians = cached_gaussians

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample['teacher_gaussians'] = self.cached_gaussians[idx]
        return sample


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

        # 防御方法：geotrap / naive_unlearning
        self.method = self.defense_config.get('method', 'geotrap')

        # 组件
        self.model_mgr = None
        self.data_mgr = None
        self.optimizer = None
        self.device = None

        # 陷阱损失（仅 geotrap 需要）
        self.trap_losses = {}
        if self.method == 'geotrap':
            self._setup_trap_losses()

        # 互锁配置（仅 geotrap 需要）
        if self.method == 'geotrap':
            coupling_config = self.defense_config.get('coupling', {})
            self.use_multiplicative = coupling_config.get('multiplicative', False)
            gc_config = coupling_config.get('gradient_conflict', {})
            self.use_gradient_conflict = gc_config.get('enabled', False)
            self.use_alternating_antialign = gc_config.get('alternating_antialign', False)
            self.lambda_conflict = gc_config.get('weight', 0.1)
            self.conflict_every_k = gc_config.get('every_k_steps', 10)

            robust_config = self.defense_config.get('robustness', {})
            self.use_param_noise = robust_config.get('enabled', False)
            self.lambda_robust = robust_config.get('weight', 0.1)
            self.noise_scale = robust_config.get('noise_scale', 0.01)
            self.robust_every_k = robust_config.get('every_k_steps', 10)
        else:
            self.use_multiplicative = False
            self.use_gradient_conflict = False
            self.use_alternating_antialign = False
            self.lambda_conflict = 0
            self.conflict_every_k = 1
            self.use_param_noise = False
            self.lambda_robust = 0
            self.noise_scale = 0
            self.robust_every_k = 1

        self._trap_step_counter = 0
        self._robust_step_counter = 0

        # 梯度累积配置
        self.gradient_accumulation_steps = self.defense_config.get('gradient_accumulation_steps', 1)

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

        # Opacity 陷阱
        if trap_config.get('opacity', {}).get('static', False):
            self.trap_losses['opacity_static'] = OpacityCollapseLoss()

        # Rotation 陷阱
        if trap_config.get('rotation', {}).get('static', False):
            self.trap_losses['rotation_static'] = RotationAnisotropyLoss()

        # 动态敏感度损失会在训练时计算，这里只记录配置
        self.dynamic_config = {
            'position': trap_config.get('position', {}).get('dynamic', False),
            'scale': trap_config.get('scale', {}).get('dynamic', False),
            'opacity': trap_config.get('opacity', {}).get('dynamic', False),
            'rotation': trap_config.get('rotation', {}).get('dynamic', False),
        }
        self.dynamic_weights = {
            'position': trap_config.get('position', {}).get('dynamic_weight', -1.0),
            'scale': trap_config.get('scale', {}).get('dynamic_weight', 0.0),
            'opacity': trap_config.get('opacity', {}).get('dynamic_weight', -1.0),
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
        print(f"  防御方法: {self.method}")
        print("=" * 80)

        self.device = device or self.config['model'].get('device', 'cuda')

        # 1. 加载学生模型（用于微调）— 防御永远不加 LoRA
        print("\n[1/5] 加载学生模型（用于微调）...")
        self.model_mgr = ModelManager(self.config)
        self.model_mgr.setup(apply_lora=False, device=self.device)
        print(f"  ✓ 学生模型已加载")

        # 2. 设置敏感层微调
        if target_layers is not None:
            print(f"\n[2/5] 设置敏感层微调...")
            self._setup_selective_finetuning(target_layers)
        else:
            print(f"\n[2/5] 跳过敏感层设置（微调所有层）")

        # 3. 设置数据加载器（双数据加载器模式）
        print("\n[3/5] 设置数据加载器...")

        # 检查配置
        data_config = self.config.get('data', {})
        if 'source' not in data_config or 'target' not in data_config:
            raise ValueError(
                "DefenseTrainer 需要同时配置 source 和 target 数据！\n"
                "请在配置文件的 data 中添加：\n"
                "  source: {categories: [...], ...}\n"
                "  target: {categories: [...], ...}"
            )

        print("  模式: 双数据加载器（Source + Target）")

        # Source数据加载器（蒸馏用）
        source_data_mgr = DataManager(self.config, self.model_mgr.opt)
        source_data_mgr.setup_dataloaders(train=True, val=False, subset='source')
        source_full_dataset = source_data_mgr.train_loader.dataset

        # Target数据加载器（defense_target：通过 object_split 自动选择 defense 物体）
        target_data_mgr = DataManager(self.config, self.model_mgr.opt)
        target_data_mgr.setup_dataloaders(train=True, val=True, subset='defense_target')
        self.target_loader = target_data_mgr.train_loader
        self.target_val_loader = target_data_mgr.val_loader

        # 混合比例
        self.source_ratio = data_config.get('source_ratio', 0.5)

        print(f"  ✓ Source数据: {len(source_full_dataset)} 样本")
        print(f"  ✓ Target数据: {len(self.target_loader.dataset)} 样本")
        print(f"  ✓ 混合比例: Source {self.source_ratio:.0%} / Target {1-self.source_ratio:.0%}")

        # 4. 预计算教师模型 Gaussian，划分 source train/val，替换 source_loader
        #    缓存命中时跳过教师模型加载，直接从磁盘读取
        print("\n[4/5] 准备蒸馏目标（Teacher Gaussians）...")
        # 预计算时用临时 loader 遍历全量 source 数据
        self.source_loader = source_data_mgr.train_loader
        cached_gaussians = self._precompute_teacher_gaussians()
        wrapped_dataset = CachedGaussianDataset(source_full_dataset, cached_gaussians)

        # 划分 source train/val（固定尾部 10% 做验证，保证可复现）
        from torch.utils.data import Subset
        n_total = len(wrapped_dataset)
        source_val_ratio = data_config.get('source_val_ratio', 0.1)
        n_val = max(int(n_total * source_val_ratio), 1)
        n_train = n_total - n_val
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_total))

        self.source_loader = DataLoader(
            Subset(wrapped_dataset, train_indices),
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
        )
        self.source_val_loader = DataLoader(
            Subset(wrapped_dataset, val_indices),
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
        )
        print(f"  ✓ Source 划分: train {n_train} / val {n_val}")

        # 5. 设置优化器
        training_config = self.config['training']
        trainable_params = [p for p in self.model_mgr.model.parameters() if p.requires_grad]
        optimizer_type = training_config.get('optimizer', 'adamw')
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=training_config['lr'],
                weight_decay=training_config['weight_decay'],
                momentum=training_config.get('optimizer_momentum', 0.9),
            )
        else:
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=training_config['lr'],
                weight_decay=training_config['weight_decay'],
                betas=tuple(training_config.get('optimizer_betas', [0.9, 0.95])),
            )

        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in self.model_mgr.model.parameters())
        print(f"\n  ✓ 可训练参数: {num_trainable:,} / {num_total:,} ({num_trainable/num_total*100:.3f}%)")

        # 打印互锁配置
        if self.use_multiplicative or self.use_gradient_conflict or self.use_param_noise:
            print(f"\n  互锁配置:")
            if self.use_multiplicative:
                print(f"    ✓ 乘法耦合: 开启")
            if self.use_gradient_conflict:
                mode = "交替反向" if self.use_alternating_antialign else "正交化"
                print(f"    ✓ 梯度冲突: 每 {self.conflict_every_k} 步, λ={self.lambda_conflict}, 模式={mode}")
            if self.use_param_noise:
                print(f"    ✓ 参数加噪: 每 {self.robust_every_k} 步, λ={self.lambda_robust}, σ={self.noise_scale}")

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

    def _precompute_teacher_gaussians(self) -> List[torch.Tensor]:
        """
        预计算教师模型在所有 source 样本上的 Gaussian 输出

        首次运行时遍历 source 数据集推理一次，结果缓存到磁盘。
        后续运行如果数据配置和模型权重未变，直接从磁盘加载，跳过教师模型推理。

        Returns:
            cached_gaussians: 每个样本的 Gaussian 参数列表，各元素 shape [N, 14]，存于 CPU
        """
        cache_path = self._get_cache_path()

        # 尝试从磁盘加载
        if cache_path.exists():
            print(f"  从缓存加载: {cache_path}")
            cached_gaussians = torch.load(cache_path, map_location='cpu', weights_only=True)
            dataset_len = len(self.source_loader.dataset)
            if len(cached_gaussians) == dataset_len:
                print(f"  ✓ 缓存命中: {len(cached_gaussians)} 个样本")
                return cached_gaussians
            print(f"  ⚠ 缓存样本数不匹配 ({len(cached_gaussians)} vs {dataset_len})，重新计算")

        # 缓存未命中，临时加载教师模型执行推理
        print("  缓存未命中，加载教师模型进行推理...")
        teacher_mgr = ModelManager(self.config)
        teacher_mgr.setup(device=self.device)
        teacher_model = teacher_mgr.model
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        dataset = self.source_loader.dataset
        temp_loader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
        )

        cached_gaussians = []
        with torch.no_grad():
            for batch in tqdm(temp_loader, desc="预计算 Teacher Gaussians"):
                input_images = batch['input_images'].to(self.device)
                gaussians = teacher_model.forward_gaussians(input_images)
                for i in range(gaussians.shape[0]):
                    cached_gaussians.append(gaussians[i].cpu())

        # 释放教师模型
        del teacher_model, teacher_mgr
        torch.cuda.empty_cache()

        # 保存到磁盘
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached_gaussians, cache_path)
        print(f"  ✓ 预计算完成: {len(cached_gaussians)} 个样本，已缓存到 {cache_path}")
        return cached_gaussians

    def _get_cache_path(self):
        """
        根据模型权重路径和 source 数据配置生成缓存文件路径

        缓存 key 包含：模型权重路径、source 数据子配置、共享数据参数。
        任何一项变化都会生成不同的 hash，触发重新计算。
        """
        from pathlib import Path

        key_dict = {
            'resume': self.config['model'].get('resume', ''),
            'source': self.config['data'].get('source', {}),
            'view_selector': self.config['data'].get('view_selector', ''),
            'angle_offset': self.config['data'].get('angle_offset', 0),
            'samples_per_object': self.config['data'].get('samples_per_object', 1),
        }
        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        digest = hashlib.sha256(key_str.encode()).hexdigest()[:12]

        cache_dir = Path(self.config.get('misc', {}).get('workspace', 'output/workspace')) / 'cache'
        return cache_dir / f"teacher_gaussians_{digest}.pt"

    def compute_distillation_loss(self, student_gaussians, teacher_gaussians):
        """
        计算蒸馏损失（保持原有能力）

        Args:
            student_gaussians: 学生模型生成的 Gaussian 参数 [B, N, 14]
            teacher_gaussians: 教师模型生成的 Gaussian 参数 [B, N, 14]

        Returns:
            loss: 蒸馏损失
        """
        order = self.defense_config.get('distill_loss_order', 2)
        diff = student_gaussians - teacher_gaussians
        return torch.mean(diff.abs() ** order)

    def compute_trap_loss(self, gaussians, model, input_images=None):
        """
        计算陷阱损失（制造几何陷阱）

        支持两种互锁机制：
        1. 乘法耦合：静态 loss 用 ∏(1 - L_i) 组合，各 trap 梯度被其他 trap 强度放大
        2. 梯度冲突正则：每 K 步惩罚不同 trap loss 梯度在权重空间的对齐度

        Args:
            gaussians: Gaussian 参数 [B, N, 14]
            model: 模型（用于计算动态敏感度和梯度冲突）

        Returns:
            loss_dict: 各项陷阱损失的字典
            total_loss: 总陷阱损失
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. 静态陷阱损失（保留 tensor 用于乘法耦合和梯度冲突）
        static_loss_tensors = {}
        for name, trap_loss_fn in self.trap_losses.items():
            loss = trap_loss_fn(gaussians)
            static_loss_tensors[name] = loss
            loss_dict[name] = loss.item()

        # 2. 组合静态损失
        if self.use_multiplicative and len(static_loss_tensors) >= 2:
            # 乘法耦合：∏(1 - L_i) - 1
            # 各 L_i < 0（最小化），所以 (1 - L_i) > 1
            # 梯度：∂/∂L_i = -∏_{j≠i}(1 - L_j)，即被其他 trap 强度放大
            product = torch.ones(1, device=gaussians.device, dtype=gaussians.dtype)
            for loss in static_loss_tensors.values():
                product = product * (1.0 - loss)
            static_combined = -(product - 1.0)
            loss_dict['static_combined'] = static_combined.item()
            total_loss += static_combined
        else:
            for loss in static_loss_tensors.values():
                total_loss += loss

        # 3. 梯度冲突正则（每 K 步）
        self._trap_step_counter += 1
        if (self.use_gradient_conflict
                and len(static_loss_tensors) >= 2
                and self._trap_step_counter % self.conflict_every_k == 0):
            conflict_loss = self._compute_gradient_conflict(
                static_loss_tensors, model
            )
            weighted_conflict = self.lambda_conflict * conflict_loss
            total_loss += weighted_conflict
            loss_dict['gradient_conflict'] = conflict_loss.item()

        # 4. 动态敏感度损失
        if self.dynamic_config['position']:
            position = gaussians[..., 0:3]
            sensitivity_loss = self._compute_sensitivity(
                position, model, self.dynamic_weights['position']
            )
            loss_dict['position_dynamic'] = sensitivity_loss.item()
            total_loss += sensitivity_loss

        if self.dynamic_config['scale']:
            scale = gaussians[..., 4:7]
            sensitivity_loss = self._compute_sensitivity(
                scale, model, self.dynamic_weights['scale']
            )
            loss_dict['scale_dynamic'] = sensitivity_loss.item()
            total_loss += sensitivity_loss

        if self.dynamic_config['opacity']:
            opacity = gaussians[..., 3:4]
            sensitivity_loss = self._compute_sensitivity(
                opacity, model, self.dynamic_weights['opacity']
            )
            loss_dict['opacity_dynamic'] = sensitivity_loss.item()
            total_loss += sensitivity_loss

        if self.dynamic_config['rotation']:
            rotation = gaussians[..., 7:11]
            sensitivity_loss = self._compute_sensitivity(
                rotation, model, self.dynamic_weights['rotation']
            )
            loss_dict['rotation_dynamic'] = sensitivity_loss.item()
            total_loss += sensitivity_loss

        # 5. 参数加噪鲁棒性（临时禁用，减少显存占用）
        # TODO: 合并为单次前向传播（直接在加噪权重上算 trap loss）
        self._robust_step_counter += 1
        # if (self.use_param_noise
        #         and len(static_loss_tensors) >= 1
        #         and self._robust_step_counter % self.robust_every_k == 0):
        #     robust_loss = self._compute_robustness_loss(
        #         input_images, static_loss_tensors, model
        #     )
        #     weighted_robust = self.lambda_robust * robust_loss
        #     total_loss += weighted_robust
        #     loss_dict['param_noise_robust'] = robust_loss.item()

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

    def _compute_gradient_conflict(self, static_loss_tensors, model):
        """
        计算逐层梯度冲突正则化（高效版本）

        只做 2 次反向传播（每个 trap 一次），然后按层提取梯度，
        在每层上独立计算余弦相似度并求和。

        支持交替反向模式：交替推动每个 trap 的梯度反向于其他 trap。

        Args:
            static_loss_tensors: {name: loss_tensor} 各静态 trap loss
            model: 模型

        Returns:
            conflict_loss: 梯度冲突损失（标量，可微分）
        """
        trap_names = list(static_loss_tensors.keys())
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        # 建立参数到层名的映射
        param_to_layer = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_to_layer[param] = name

        # 收集所有层名（去重，取到 block 级别）
        layer_names = set()
        for name in param_to_layer.values():
            # 提取到 nets.X 或 attns.X 级别
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part in ('nets', 'attns') and i + 1 < len(parts):
                    layer_names.add('.'.join(parts[:i+2]))
                    break
            else:
                # 没有 nets/attns，取前3段
                layer_names.add('.'.join(parts[:min(3, len(parts))]))
        layer_names = sorted(layer_names)

        if self.use_alternating_antialign:
            # 交替反向模式：只做 2 次反向传播
            phase = (self._trap_step_counter // self.conflict_every_k) % len(trap_names)
            active_name = trap_names[phase]
            reference_names = [n for n in trap_names if n != active_name]

            # 1 次反向传播：active trap（可微）
            # retain_graph=True 因为后面还要算 reference grads
            g_active_all = torch.autograd.grad(
                outputs=static_loss_tensors[active_name],
                inputs=trainable_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            # 1 次反向传播：reference trap（固定）
            # 必须保留计算图，因为后面还要对 total_loss 做 backward()
            g_ref_all_list = []
            for i, ref_name in enumerate(reference_names):
                g_ref = torch.autograd.grad(
                    outputs=static_loss_tensors[ref_name],
                    inputs=trainable_params,
                    create_graph=False,
                    retain_graph=True,  # 必须保留，因为 total_loss 还要用这些 tensor
                    allow_unused=True,
                )
                g_ref_all_list.append(g_ref)

            # 按层提取梯度并计算冲突
            conflict_loss_total = torch.zeros(1, device=self.device, dtype=torch.float32)
            num_layers = 0

            for layer_name in layer_names:
                # 提取这一层的参数索引
                active_grads = []
                ref_grads_list = [[] for _ in reference_names]

                for idx, param in enumerate(trainable_params):
                    pname = param_to_layer[param]
                    if pname.startswith(layer_name):
                        g_a = g_active_all[idx]
                        if g_a is not None:
                            active_grads.append(g_a.reshape(-1))
                        else:
                            active_grads.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))

                        for ri, g_ref_all in enumerate(g_ref_all_list):
                            g_r = g_ref_all[idx]
                            if g_r is not None:
                                ref_grads_list[ri].append(g_r.reshape(-1).detach())
                            else:
                                ref_grads_list[ri].append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))

                if len(active_grads) == 0:
                    continue

                g_active_vec = torch.cat(active_grads)

                for ref_grads in ref_grads_list:
                    g_ref_vec = torch.cat(ref_grads)
                    norm_a = g_active_vec.norm() + 1e-8
                    norm_r = g_ref_vec.norm() + 1e-8
                    cos_sim = torch.dot(g_active_vec, g_ref_vec) / (norm_a * norm_r)
                    conflict_loss_total = conflict_loss_total + cos_sim

                num_layers += 1

            if num_layers > 0:
                conflict_loss_total = conflict_loss_total / num_layers

            return conflict_loss_total.squeeze()

        else:
            # 正交化模式：每个 trap 做 1 次反向传播
            # 必须保留计算图，因为后面还要对 total_loss 做 backward()
            all_grads = {}
            trap_items = list(static_loss_tensors.items())
            for i, (trap_name, loss) in enumerate(trap_items):
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=trainable_params,
                    create_graph=True,
                    retain_graph=True,  # 必须保留，因为 total_loss 还要用这些 tensor
                    allow_unused=True,
                )
                all_grads[trap_name] = grads

            # 按层提取梯度并计算冲突
            conflict_loss_total = torch.zeros(1, device=self.device, dtype=torch.float32)
            num_layers = 0

            for layer_name in layer_names:
                layer_grad_vecs = {}
                for trap_name, grads in all_grads.items():
                    layer_grads = []
                    for idx, param in enumerate(trainable_params):
                        pname = param_to_layer[param]
                        if pname.startswith(layer_name):
                            g = grads[idx]
                            if g is not None:
                                layer_grads.append(g.reshape(-1))
                            else:
                                layer_grads.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))

                    if len(layer_grads) > 0:
                        layer_grad_vecs[trap_name] = torch.cat(layer_grads)

                if len(layer_grad_vecs) < 2:
                    continue

                names = list(layer_grad_vecs.keys())
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        g_i = layer_grad_vecs[names[i]]
                        g_j = layer_grad_vecs[names[j]]
                        cos_sim = torch.dot(g_i, g_j) / (g_i.norm() * g_j.norm() + 1e-8)
                        conflict_loss_total = conflict_loss_total + torch.relu(cos_sim)

                num_layers += 1

            if num_layers > 0:
                conflict_loss_total = conflict_loss_total / num_layers

            return conflict_loss_total.squeeze()
    def _compute_robustness_loss(self, input_images, static_loss_tensors, model):
        """
        计算参数加噪鲁棒性损失

        对所有模型参数添加高斯噪声，计算噪声权重下的 trap loss，
        确保陷阱对权重扰动鲁棒（模拟攻击者的 full fine-tuning）。

        Args:
            input_images: 输入图像 [B, 4, 9, H, W]
            static_loss_tensors: {name: loss_tensor} 各静态 trap loss（干净权重下）
            model: 模型

        Returns:
            robustness_loss: 鲁棒性损失（标量，可微分）
        """
        # 1. 保存所有参数的原始权重
        original_state = {}
        for name, param in model.named_parameters():
            original_state[name] = param.data.clone()

        # 2. 对所有参数添加高斯噪声（模拟攻击者的 full fine-tuning）
        for name, param in model.named_parameters():
            noise = torch.randn_like(param) * self.noise_scale
            param.data.add_(noise)

        # 3. 计算噪声权重下的 trap loss
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gaussians_noisy = model.forward_gaussians(input_images)
        gaussians_noisy = gaussians_noisy.float()

        trap_loss_noisy = torch.zeros(1, device=gaussians_noisy.device, dtype=gaussians_noisy.dtype)
        for name, trap_loss_fn in self.trap_losses.items():
            loss_noisy = trap_loss_fn(gaussians_noisy)
            trap_loss_noisy = trap_loss_noisy + loss_noisy

        # 4. 恢复所有参数的原始权重
        for name, param in model.named_parameters():
            param.data.copy_(original_state[name])

        # 5. 鲁棒性损失：希望噪声下 trap 仍然强（trap loss 是负值，越负越强）
        # 最小化 -trap_loss_noisy → 最大化 trap_loss_noisy（更负）
        robustness_loss = -trap_loss_noisy

        return robustness_loss.squeeze()

    def train_step(self, batch, is_target_data=True):
        """
        训练一个 step（只计算损失和反向传播，不更新参数）

        Args:
            batch: 数据批次
            is_target_data: 是否为 target 数据（True=陷阱/遗忘损失，False=蒸馏损失）

        Returns:
            loss_dict: 损失字典
        """
        self.model_mgr.model.train()

        # 移动数据到设备
        input_images = batch['input_images'].to(self.device)  # [B, 4, 9, H, W]

        # 学生模型前向传播（保持 bf16 以节省显存）
        with torch.set_grad_enabled(True):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                student_gaussians = self.model_mgr.model.forward_gaussians(input_images)
                # 保持 bf16，不转换为 fp32

        loss_dict = {}

        # 所有 loss 计算在 bf16 下进行以节省显存
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if is_target_data:
                if self.method == 'naive_unlearning':
                    # Naive Unlearning: 对 target 数据的渲染 loss 做梯度上升
                    total_loss = self._compute_naive_unlearning_loss(batch, loss_dict)
                else:
                    # GeoTrap: 计算陷阱损失
                    trap_loss_dict, trap_loss = self.compute_trap_loss(student_gaussians, self.model_mgr.model, input_images)
                    loss_dict.update(trap_loss_dict)

                    lambda_trap = self.defense_config.get('lambda_trap', 1.0)
                    total_loss = lambda_trap * trap_loss

            else:
                # Source Data: 使用预计算的教师 Gaussian 计算蒸馏损失
                teacher_gaussians = batch['teacher_gaussians'].to(self.device)

                distill_loss = self.compute_distillation_loss(student_gaussians, teacher_gaussians)
                loss_dict['distillation'] = distill_loss.item()

                lambda_distill = self.defense_config.get('lambda_distill', 1.0)
                total_loss = lambda_distill * distill_loss

        loss_dict['loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        # 反向传播（梯度累积，不立即更新参数）
        if isinstance(total_loss, torch.Tensor):
            # 梯度缩放：除以累积步数，使得累积后的梯度大小正确
            scaled_loss = total_loss / self.gradient_accumulation_steps
            scaled_loss.backward()

        return loss_dict

    def _compute_naive_unlearning_loss(self, batch, loss_dict):
        """
        计算 Naive Unlearning 损失：对 target 数据的渲染 loss 做梯度上升

        通过 prepare_lgm_data 准备数据，调用 model.forward() 获取渲染损失，
        然后取负（梯度上升），使模型主动忘掉如何渲染 target 物体。

        Args:
            batch: 数据批次
            loss_dict: 损失字典（会被原地更新）

        Returns:
            total_loss: 取负后的渲染损失（用于梯度上升）
        """
        from tools.utils import prepare_lgm_data

        data = prepare_lgm_data(batch, self.model_mgr.model, self.device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            results = self.model_mgr.model.forward(data, step_ratio=1.0)

        render_loss = results['loss']
        loss_dict['render_loss'] = render_loss.item()

        # 取负：最小化 -render_loss = 最大化 render_loss（梯度上升）
        lambda_trap = self.defense_config.get('lambda_trap', 1.0)
        total_loss = -lambda_trap * render_loss

        loss_dict['unlearning_loss'] = total_loss.item()
        return total_loss

    def train_epoch(self, epoch: int, global_step: int = 0, step_callback=None):
        """
        训练一个 epoch（双数据加载器模式 + 梯度累积）

        Args:
            epoch: 当前 epoch 编号
            global_step: 全局步数起始值（跨 epoch 累计）
            step_callback: 每步回调 callable(global_step, loss_dict) -> None
                           可用于周期性评估等外部逻辑

        Returns:
            avg_metrics: 平均损失字典
            global_step: 更新后的全局步数
        """
        self.model_mgr.model.train()

        total_losses = {}
        num_batches = 0
        accumulation_counter = 0  # 梯度累积计数器

        # 双数据加载器：按比例混合source和target
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        # 计算总batch数（按 target 数据量定义 epoch，source 是辅助）
        max_batches = len(self.target_loader)

        # 预先计算每个梯度累积周期的 source/target 分配
        # 例如：source_ratio=0.8, gradient_accumulation_steps=8 → 每周期 6 个 source + 2 个 target
        num_source_per_cycle = int(self.gradient_accumulation_steps * self.source_ratio)
        num_target_per_cycle = self.gradient_accumulation_steps - num_source_per_cycle

        pbar = tqdm(range(max_batches), desc=f"Epoch {epoch}")
        for batch_idx in pbar:
            # 在每个梯度累积周期开始时，生成固定的 source/target 序列
            if accumulation_counter == 0:
                # 生成一个包含 num_source_per_cycle 个 False 和 num_target_per_cycle 个 True 的列表
                cycle_schedule = [False] * num_source_per_cycle + [True] * num_target_per_cycle
                # 打乱顺序，但保证每个周期的组成是固定的
                import random
                random.shuffle(cycle_schedule)

            # 根据预先分配的序列决定使用 source 还是 target
            use_target = cycle_schedule[accumulation_counter]

            try:
                if use_target:
                    batch = next(target_iter)
                    loss_dict = self.train_step(batch, is_target_data=True)
                else:
                    batch = next(source_iter)
                    loss_dict = self.train_step(batch, is_target_data=False)

            except StopIteration:
                if use_target:
                    target_iter = iter(self.target_loader)
                    batch = next(target_iter)
                    loss_dict = self.train_step(batch, is_target_data=True)
                else:
                    source_iter = iter(self.source_loader)
                    batch = next(source_iter)
                    loss_dict = self.train_step(batch, is_target_data=False)

            # 累积损失
            for key, value in loss_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            num_batches += 1
            accumulation_counter += 1

            # 梯度累积：每 N 步或最后一个 batch 时更新参数
            if accumulation_counter % self.gradient_accumulation_steps == 0 or batch_idx == max_batches - 1:
                # 梯度裁剪
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_mgr.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                # 更新参数
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulation_counter = 0
                # 释放 CUDA 缓存碎片
                torch.cuda.empty_cache()

            global_step += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'step': global_step,
                'accum': f"{accumulation_counter}/{self.gradient_accumulation_steps}",
            })

            # 步回调
            if step_callback is not None:
                step_callback(global_step, loss_dict)
                # 回调可能切换模型到 eval 模式，恢复 train
                self.model_mgr.model.train()

        # 计算平均损失
        avg_metrics = {
            key: total / num_batches if num_batches > 0 else 0
            for key, total in total_losses.items()
        }

        return avg_metrics, global_step

    def validate(self):
        """
        验证模型

        同时评估：
        1. Target 数据上的效果（geotrap: trap 损失 / naive_unlearning: 渲染损失）
        2. Source 数据上的蒸馏质量（与预计算 teacher Gaussian 的 MSE）

        Returns:
            avg_metrics: 验证损失字典
        """
        self.model_mgr.model.eval()

        total_losses = {}
        num_batches = 0

        # 1. Target 验证
        if self.method == 'naive_unlearning':
            # Naive Unlearning: 计算渲染损失（越高 = 遗忘越彻底）
            from tools.utils import prepare_lgm_data
            with torch.no_grad():
                for batch in tqdm(self.target_val_loader, desc="Val [Target]"):
                    data = prepare_lgm_data(batch, self.model_mgr.model, self.device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        results = self.model_mgr.model.forward(data, step_ratio=1.0)
                    total_losses.setdefault('render_loss', 0.0)
                    total_losses['render_loss'] += results['loss'].item()
                    num_batches += 1
        else:
            # GeoTrap: trap 效果
            with torch.no_grad():
                for batch in tqdm(self.target_val_loader, desc="Val [Target]"):
                    input_images = batch['input_images'].to(self.device)
                    student_gaussians = self.model_mgr.model.forward_gaussians(input_images)

                    loss_dict = {}
                    for name, trap_loss_fn in self.trap_losses.items():
                        loss = trap_loss_fn(student_gaussians)
                        loss_dict[name] = loss.item()

                    for key, value in loss_dict.items():
                        if key not in total_losses:
                            total_losses[key] = 0.0
                        total_losses[key] += value
                    num_batches += 1

        # 2. Source 验证：蒸馏质量
        source_mse_total = 0.0
        source_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.source_val_loader, desc="Val [Source]"):
                input_images = batch['input_images'].to(self.device)
                teacher_gaussians = batch['teacher_gaussians'].to(self.device)
                student_gaussians = self.model_mgr.model.forward_gaussians(input_images)
                mse = torch.nn.functional.mse_loss(student_gaussians, teacher_gaussians)
                source_mse_total += mse.item()
                source_batches += 1

        # 合并指标
        avg_metrics = {
            key: total / num_batches if num_batches > 0 else 0
            for key, total in total_losses.items()
        }
        avg_metrics['source_distill_mse'] = (
            source_mse_total / source_batches if source_batches > 0 else 0
        )

        # 3. 耦合与冲突指标（从平均 trap loss 计算）
        trap_names = list(self.trap_losses.keys())
        if len(trap_names) >= 2:
            l1 = avg_metrics[trap_names[0]]
            l2 = avg_metrics[trap_names[1]]
            avg_metrics['coupling_value'] = -((1 - l1) * (1 - l2) - 1)

        # 梯度余弦相似度：取一个 target val batch 计算（不需要 create_graph）
        if len(trap_names) >= 2:
            avg_metrics['grad_cosine_sim'] = self._compute_val_grad_cosine()

        # 验证结束后清理 CUDA 缓存
        torch.cuda.empty_cache()

        return avg_metrics

    def _compute_val_grad_cosine(self) -> float:
        """
        在一个 target val batch 上计算两个 trap loss 梯度的余弦相似度

        不需要 create_graph，只做一阶梯度计算。
        返回值 ∈ [-1, 1]：0 = 正交，1 = 对齐，-1 = 相反。
        """
        self.model_mgr.model.eval()
        batch = next(iter(self.target_val_loader))
        input_images = batch['input_images'].to(self.device)

        # 需要梯度来计算 grad
        with torch.enable_grad():
            student_gaussians = self.model_mgr.model.forward_gaussians(input_images)

            trainable_params = [p for p in self.model_mgr.model.parameters() if p.requires_grad]
            grad_vectors = []

            trap_items = list(self.trap_losses.items())
            for i, (name, trap_loss_fn) in enumerate(trap_items):
                is_last = (i == len(trap_items) - 1)
                loss = trap_loss_fn(student_gaussians)
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=trainable_params,
                    retain_graph=not is_last,
                    allow_unused=True,
                )
                grad_vec = torch.cat([
                    g.reshape(-1) if g is not None
                    else torch.zeros(p.numel(), device=p.device)
                    for g, p in zip(grads, trainable_params)
                ])
                grad_vectors.append(grad_vec)

        g1, g2 = grad_vectors[0], grad_vectors[1]
        cos_sim = torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-8)
        result = cos_sim.item()

        # 显式释放计算图和中间张量
        del grad_vectors, g1, g2, cos_sim, student_gaussians, input_images, batch
        torch.cuda.empty_cache()

        return result

    def train(self, num_epochs: int, save_dir: str = None, validate_every: int = 1,
              step_callback=None):
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数
            save_dir: 保存目录（None=不保存）
            validate_every: 每隔多少个 epoch 验证一次
            step_callback: 每步回调 callable(global_step, loss_dict) -> None
        """
        print("\n" + "=" * 80)
        print(f"开始防御训练 - {num_epochs} epochs")
        print("=" * 80)

        global_step = 0
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics, global_step = self.train_epoch(
                epoch, global_step=global_step, step_callback=step_callback)
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
        if epoch == self.config['training'].get('defense_epochs', self.config['training'].get('num_epochs', 10)):
            final_path = os.path.join(save_dir, "model_defense.pth")
            torch.save(self.model_mgr.model.state_dict(), final_path)
            print(f"[DefenseTrainer] 保存最终模型: {final_path}")

            # 自动注册到模型标签仓库
            tag = self.defense_config.get('tag')
            if tag:
                metadata = {
                    "epoch": epoch,
                    "target_layers": self.target_layers,
                    "trap_losses": list(self.trap_losses.keys()),
                    "source_path": final_path,
                    "defense_config": {
                        k: v for k, v in self.defense_config.items()
                        if k not in ('target_data',)
                    },
                }
                registry_register(tag, final_path, metadata=metadata)
            else:
                print(f"[DefenseTrainer] 提示: 未配置 defense.tag，跳过注册。"
                      f"可在 config 中添加 defense.tag 自动注册模型")


def load_or_train_defense(config, device='cuda', save_dir=None):
    """
    一行加载或训练防御模型。

    根据配置自动计算 hash tag，如果 registry 中已存在则跳过训练，否则触发训练并注册。
    defense.method='none' 时跳过防御，返回 None。

    Args:
        config: 完整配置字典（会被修改 defense.tag）
        device: 训练设备
        save_dir: checkpoint 保存目录（默认自动生成）

    Returns:
        (tag, defense_history):
            tag — hash 字符串，用于 model_resume_override=f"tag:{tag}"；method='none' 时为 None
            defense_history — 训练历史列表，缓存命中时为 None
    """
    import copy
    from tools.utils import compute_defense_hash
    from tools.model_registry import REGISTRY_DIR

    method = config.get('defense', {}).get('method', 'geotrap')
    if method == 'none':
        print("[Defense] method=none，跳过防御训练")
        return None, None

    config = copy.deepcopy(config)
    tag = compute_defense_hash(config)
    model_path = REGISTRY_DIR / tag / "model.pth"

    if model_path.exists():
        print(f"[Defense] 缓存命中: tag={tag}")
        print(f"[Defense] 模型路径: {model_path}")
        return tag, None

    print(f"[Defense] 缓存未命中: tag={tag}，开始训练...")
    config['defense']['tag'] = tag

    target_layers = config.get('defense', {}).get('target_layers')
    defense_epochs = config['training'].get('defense_epochs', 25)

    if save_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        workspace = config.get('misc', {}).get('workspace', './output/workspace')
        save_dir = os.path.join(workspace, f"defense_{tag}_{timestamp}")

    trainer = DefenseTrainer(config)
    trainer.setup(device=device, target_layers=target_layers)

    epoch_history = []
    global_step = 0
    validate_every = 5

    for epoch in range(1, defense_epochs + 1):
        train_metrics, global_step = trainer.train_epoch(epoch, global_step)
        combined = {f"train_{k}": v for k, v in train_metrics.items()}
        combined['epoch'] = epoch

        do_val = (epoch % validate_every == 0) or (epoch == defense_epochs)
        if do_val:
            val_metrics = trainer.validate()
            combined.update({f"val_{k}": v for k, v in val_metrics.items()})
            print(f"  [Defense] Epoch {epoch}/{defense_epochs} - "
                  f"Loss: {train_metrics['loss']:.4f}, "
                  f"DistillMSE: {val_metrics.get('source_distill_mse', 0):.6f}")
            for k in ('position_static', 'scale_static', 'opacity_static',
                      'coupling_value', 'grad_cosine_sim'):
                if k in val_metrics:
                    print(f"    {k}: {val_metrics[k]:.4f}")
        else:
            print(f"  [Defense] Epoch {epoch}/{defense_epochs} - "
                  f"Loss: {train_metrics['loss']:.4f}")

        epoch_history.append(combined)

        if epoch == defense_epochs:
            trainer.save_checkpoint(save_dir, epoch)

    del trainer
    torch.cuda.empty_cache()

    return tag, epoch_history

