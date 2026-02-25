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
from tools.utils import prepare_lgm_data


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
        optimizer_type: str = 'adamw',
        optimizer_betas: tuple = (0.9, 0.95),
        optimizer_momentum: float = 0.9,
        include_input_supervision: bool = True,
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
            optimizer_type: 优化器类型 ('adamw' 或 'sgd')
            optimizer_betas: AdamW 的 betas 参数
            optimizer_momentum: SGD 的 momentum 参数
            include_input_supervision: 是否将输入视图纳入监督。
                True（默认）: 标准攻击，输入视图参与 loss
                False: 语义偏转攻击，输入视图不参与 loss
        """
        self.model = model
        self.device = device
        self.gradient_clip = gradient_clip
        self.lambda_lpips = lambda_lpips
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.include_input_supervision = include_input_supervision

        # 混合精度：兼容旧的 bool 参数
        if isinstance(mixed_precision, bool):
            self.mixed_precision = 'fp16' if mixed_precision else 'no'
        else:
            self.mixed_precision = mixed_precision

        # 优化器
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=optimizer_momentum,
            )
        else:
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(optimizer_betas),
            )

        # GradScaler 只有 fp16 需要，bf16 不需要
        self.scaler = GradScaler('cuda') if self.mixed_precision == 'fp16' else None

        # 调试：打印优化器参数组信息
        num_param_groups = len(self.optimizer.param_groups)
        total_params_in_optimizer = sum(len(pg['params']) for pg in self.optimizer.param_groups)
        print(f"  [DEBUG] Optimizer: {optimizer_type}, lr={lr}, "
              f"param_groups={num_param_groups}, "
              f"total_params_in_optimizer={total_params_in_optimizer}, "
              f"mixed_precision={self.mixed_precision}, "
              f"grad_accum={gradient_accumulation_steps}")

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
        准备LGM模型需要的数据格式（委托给 prepare_lgm_data）。

        Args:
            batch: 数据加载器返回的批次

        Returns:
            data: LGM模型期望的数据字典
        """
        return prepare_lgm_data(batch, self.model, self.device,
                                include_input_supervision=self.include_input_supervision)

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

        # Masked 指标：只在物体区域计算，去掉白色背景的稀释
        with torch.no_grad():
            pred_images = results.get('images_pred')
            pred_alphas = results.get('alphas_pred')
            gt_images = data['images_output']
            gt_masks = data['masks_output']

            if pred_images is not None and gt_masks is not None:
                # gt_images 已被 model.forward() 合成白色背景，pred_images 也是白色背景
                # mask 区域内计算 MSE → masked PSNR
                mask_flat = gt_masks.reshape(-1)
                pred_flat = pred_images.reshape(-1, 3)
                gt_flat = gt_images.reshape(-1, 3)
                mask_sum = mask_flat.sum().clamp(min=1.0)
                masked_mse = ((pred_flat - gt_flat) ** 2 * mask_flat.unsqueeze(-1)).sum() / (mask_sum * 3)
                masked_psnr = -10 * torch.log10(masked_mse + 1e-8)
                loss_dict['masked_psnr'] = masked_psnr.item()

                # Masked LPIPS：裁剪物体区域计算LPIPS
                if hasattr(self._raw_model, 'lpips_loss'):
                    import torch.nn.functional as F
                    B, V, C, H, W = pred_images.shape

                    # 对每个样本计算masked LPIPS
                    masked_lpips_list = []
                    for b in range(B):
                        for v in range(V):
                            mask_v = gt_masks[b, v, 0]  # [H, W]
                            if mask_v.sum() < 10:  # 跳过物体太小的视角
                                continue

                            # 找到物体的bounding box
                            rows = mask_v.sum(dim=1)
                            cols = mask_v.sum(dim=0)
                            row_indices = (rows > 0).nonzero(as_tuple=True)[0]
                            col_indices = (cols > 0).nonzero(as_tuple=True)[0]

                            if len(row_indices) > 0 and len(col_indices) > 0:
                                y1, y2 = row_indices[0].item(), row_indices[-1].item() + 1
                                x1, x2 = col_indices[0].item(), col_indices[-1].item() + 1

                                # 裁剪物体区域
                                gt_crop = gt_images[b, v:v+1, :, y1:y2, x1:x2]
                                pred_crop = pred_images[b, v:v+1, :, y1:y2, x1:x2]

                                # 调整到256x256
                                gt_crop_256 = F.interpolate(gt_crop * 2 - 1, (256, 256), mode='bilinear', align_corners=False)
                                pred_crop_256 = F.interpolate(pred_crop * 2 - 1, (256, 256), mode='bilinear', align_corners=False)

                                # 计算LPIPS
                                lpips_crop = self._raw_model.lpips_loss(gt_crop_256, pred_crop_256)
                                masked_lpips_list.append(lpips_crop.item())

                    if len(masked_lpips_list) > 0:
                        loss_dict['masked_lpips'] = sum(masked_lpips_list) / len(masked_lpips_list)
                    else:
                        loss_dict['masked_lpips'] = 0.0
                else:
                    loss_dict['masked_lpips'] = 0.0

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
        total_masked_psnr = 0
        num_batches = 0
        num_updates = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            loss_dict, updated = self.train_step(batch)

            total_loss += loss_dict['loss']
            total_loss_lpips += loss_dict.get('loss_lpips', 0)
            total_psnr += loss_dict.get('psnr', 0)
            total_masked_psnr += loss_dict.get('masked_psnr', 0)
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
            'masked_psnr': total_masked_psnr / num_batches,
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


def run_attack(config, target_train_loader, source_val_loader,
               supervision_loader=None, target_eval_loader=None,
               save_dir=None, attack_epochs=None,
               num_render=3, eval_every_steps=10,
               model_resume_override=None, phase_name="Attack",
               return_gaussians=False, ref_gaussians=None):
    """
    一行运行攻击阶段。

    加载模型 → LoRA 微调 → 周期性评估 → 渲染样本 → 清理 GPU。

    标准模式：target_train_loader 同时用于训练和评估。
    语义偏转模式：target_train_loader 是配对数据集（input=A, supervision=B），
                 supervision_loader 用于跨类别评估 GT，
                 target_eval_loader 用于 vs input 评估。

    Args:
        config: 完整配置字典
        target_train_loader: 训练数据加载器。标准模式=target数据，语义偏转=配对数据集
        source_val_loader: source 验证数据加载器
        supervision_loader: 监督类别数据加载器（语义偏转评估用），None=标准模式
        target_eval_loader: 输入类别数据加载器（语义偏转评估 vs input 用）
        save_dir: 渲染结果保存目录
        attack_epochs: 攻击 epoch 数（默认从 config 读取）
        num_render: 渲染样本数
        eval_every_steps: 每隔多少 step 评估一次
        model_resume_override: 覆盖 model.resume（如 "tag:xxx"）
        phase_name: 阶段名称（用于日志）
        return_gaussians: 是否返回攻击后在 target 上生成的 Gaussian 列表
        ref_gaussians: 参考 Gaussian 列表（如 baseline 缓存），用于计算距离

    Returns:
        (step_history, source_metrics, target_metrics) 或
        (step_history, source_metrics, target_metrics, gaussians_list)（当 return_gaussians=True）
            step_history — [{step, epoch, loss, lpips, masked_lpips, ...}, ...]
            source_metrics — 攻击前 source 指标 {psnr, lpips}（masked）
            target_metrics — 攻击后 target 指标，标准模式: {psnr, lpips}，
                           语义偏转模式: {'input': {psnr, lpips}, 'supervision': {psnr, lpips}}
    """
    import copy
    from models import ModelManager
    from evaluation import Evaluator

    if attack_epochs is None:
        attack_epochs = config['training'].get('attack_epochs', 5)

    is_semantic_deflection = (supervision_loader is not None)

    print(f"\n{'='*80}")
    print(f"  {phase_name}")
    print(f"  Attack epochs: {attack_epochs}, eval_every_steps: {eval_every_steps}")
    if is_semantic_deflection:
        print(f"  模式: 语义偏转攻击")
        print(f"  输入数据: {len(target_train_loader.dataset)} 样本")
        print(f"  监督数据: {len(supervision_loader.dataset)} 样本")
    else:
        print(f"  模式: 标准攻击")
    print(f"{'='*80}")

    attack_config = copy.deepcopy(config)
    if model_resume_override:
        attack_config['model']['resume'] = model_resume_override
    model_mgr = ModelManager(attack_config)
    model_mgr.setup(device='cuda')
    model = model_mgr.model

    training_cfg = config['training']
    print(f"  [run_attack] 实际使用的训练参数: lr={training_cfg['lr']}, "
          f"optimizer={training_cfg.get('optimizer', 'adamw')}, "
          f"mode={training_cfg.get('mode', 'full')}")
    finetuner = AutoFineTuner(
        model=model, device='cuda',
        lr=training_cfg['lr'],
        weight_decay=training_cfg['weight_decay'],
        gradient_clip=training_cfg['gradient_clip'],
        mixed_precision=training_cfg.get('mixed_precision', 'bf16'),
        lambda_lpips=training_cfg.get('lambda_lpips', 1.0),
        gradient_accumulation_steps=training_cfg['gradient_accumulation_steps'],
        optimizer_type=training_cfg.get('optimizer', 'adamw'),
        optimizer_betas=training_cfg.get('optimizer_betas', [0.9, 0.95]),
        optimizer_momentum=training_cfg.get('optimizer_momentum', 0.9),
        include_input_supervision=not is_semantic_deflection,
    )

    evaluator = Evaluator(model, device='cuda')
    os.makedirs(save_dir, exist_ok=True)

    # 攻击前评估 source（LoRA 模式下禁用 adapter 测底座能力）
    print(f"  评估攻击前的 source 质量...")
    has_lora = hasattr(model, 'disable_adapter_layers')
    if has_lora:
        model.disable_adapter_layers()
    source_metrics = evaluator.evaluate_on_loader(source_val_loader)
    if has_lora:
        model.enable_adapter_layers()
    print(f"  攻击前 Source PSNR: {source_metrics['psnr']:.2f}, "
          f"LPIPS: {source_metrics['lpips']:.4f}")

    evaluator.render_samples(source_val_loader,
                             os.path.join(save_dir, 'source_renders'),
                             prefix='source_', num_samples=num_render)

    step_history = []
    global_step = 0
    interval_loss, interval_lpips, interval_masked_psnr, interval_masked_lpips = 0, 0, 0, 0
    interval_count = 0
    total_steps = attack_epochs * len(target_train_loader)

    # 调试：训练前参数校验
    param_sum_before = sum(p.data.sum().item() for p in model.parameters())
    print(f"  [DEBUG] 训练前参数 sum: {param_sum_before:.6f}")

    for epoch in range(1, attack_epochs + 1):
        model.train()
        for batch in target_train_loader:
            loss_dict, updated = finetuner.train_step(batch)
            global_step += 1

            interval_loss += loss_dict['loss']
            interval_lpips += loss_dict.get('loss_lpips', 0)
            interval_masked_psnr += loss_dict.get('masked_psnr', 0)
            interval_masked_lpips += loss_dict.get('masked_lpips', 0)
            interval_count += 1

            if global_step % eval_every_steps == 0 or global_step == total_steps:
                metrics = {
                    'step': global_step,
                    'epoch': epoch,
                    'loss': interval_loss / interval_count,
                    'lpips': interval_lpips / interval_count,
                    'masked_lpips': interval_masked_lpips / interval_count,
                    'masked_psnr': interval_masked_psnr / interval_count,
                }
                step_history.append(metrics)

                print(f"  [{phase_name}] Step {global_step}/{total_steps} (Ep{epoch}) - "
                      f"Loss: {metrics['loss']:.4f}, "
                      f"LPIPS: {metrics['masked_lpips']:.4f}, "
                      f"PSNR: {metrics['masked_psnr']:.2f}")

                interval_loss, interval_lpips, interval_masked_psnr, interval_masked_lpips = 0, 0, 0, 0
                interval_count = 0

        # epoch 结束，flush 残余梯度
        if finetuner._accumulation_counter % finetuner.gradient_accumulation_steps != 0:
            if finetuner.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), finetuner.gradient_clip)
            finetuner.optimizer.step()
            finetuner.optimizer.zero_grad()

    # 调试：训练后参数校验
    param_sum_after = sum(p.data.sum().item() for p in model.parameters())
    print(f"  [DEBUG] 训练后参数 sum: {param_sum_after:.6f}")
    print(f"  [DEBUG] 参数变化量: {param_sum_after - param_sum_before:.6f}")
    print(f"  [DEBUG] evaluator.model is model: {evaluator.model is model}")

    # 攻击后评估和渲染
    if is_semantic_deflection:
        # 语义偏转模式：
        # vs Input: 输入 coconut → 输出 vs coconut GT（标准评估）
        # vs Supervision: 输入 coconut → 从 durian 视角渲染 → vs durian GT
        print(f"  评估攻击后的质量...")

        # vs 输入类别：输入 coconut，与 coconut GT 对比
        eval_loader = target_eval_loader or target_train_loader
        evaluator.render_samples(eval_loader,
                                 os.path.join(save_dir, 'input_renders'),
                                 prefix='input_', num_samples=num_render)
        input_metrics = evaluator.evaluate_on_loader(eval_loader)
        print(f"  vs Input PSNR: {input_metrics['psnr']:.2f}, "
              f"LPIPS: {input_metrics['lpips']:.4f}")

        # vs 监督类别：输入 coconut → 从 durian 视角渲染 → 与 durian GT 对比
        supervision_metrics = evaluator.evaluate_cross_category(
            eval_loader, supervision_loader)
        print(f"  vs Supervision PSNR: {supervision_metrics['psnr']:.2f}, "
              f"LPIPS: {supervision_metrics['lpips']:.4f}")

        target_metrics = {
            'input': input_metrics,
            'supervision': supervision_metrics,
        }
    else:
        # 标准模式：原有逻辑
        evaluator.render_samples(target_train_loader,
                                 os.path.join(save_dir, 'target_renders'),
                                 prefix='target_', num_samples=num_render)

        print(f"  评估攻击后的 target 质量...")
        target_metrics = evaluator.evaluate_on_loader(target_train_loader)
        print(f"  攻击后 Target PSNR: {target_metrics['psnr']:.2f}, "
              f"LPIPS: {target_metrics['lpips']:.4f}")

    # Gaussian 诊断（标准模式和语义偏转模式都做）
    print(f"  Gaussian 诊断...")
    eval_loader_for_diag = target_eval_loader if is_semantic_deflection else target_train_loader
    diag_result = evaluator.diagnose_gaussians(
        eval_loader_for_diag, num_samples=8,
        return_gaussians=return_gaussians,
        ref_gaussians=ref_gaussians,
    )
    if return_gaussians:
        gaussian_diag, gaussians_list = diag_result
    else:
        gaussian_diag = diag_result
        gaussians_list = None

    print(f"  诊断结果: {gaussian_diag['diagnosis']}")
    print(f"    opacity_mean={gaussian_diag['opacity_mean']:.4f}, "
          f"pos_spread={gaussian_diag['pos_spread']:.4f}, "
          f"scale_mean={gaussian_diag['scale_mean']:.6f}, "
          f"render_white={gaussian_diag['render_white_ratio']:.4f}")
    print(f"    trap: position={gaussian_diag['trap_position']:.4f}, "
          f"scale={gaussian_diag['trap_scale']:.4f}, "
          f"opacity={gaussian_diag['trap_opacity']:.4f}, "
          f"rotation={gaussian_diag['trap_rotation']:.4f}")
    if 'gaussian_dist_to_baseline' in gaussian_diag:
        print(f"    gaussian_dist_to_baseline={gaussian_diag['gaussian_dist_to_baseline']:.6f}")
    if isinstance(target_metrics, dict) and 'input' in target_metrics:
        target_metrics['gaussian_diag'] = gaussian_diag
    else:
        target_metrics['gaussian_diag'] = gaussian_diag

    del finetuner, evaluator, model, model_mgr
    torch.cuda.empty_cache()

    if return_gaussians:
        return step_history, source_metrics, target_metrics, gaussians_list
    return step_history, source_metrics, target_metrics