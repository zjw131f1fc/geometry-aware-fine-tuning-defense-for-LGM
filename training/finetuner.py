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
               save_dir=None, attack_epochs=None, attack_steps=None,
               num_render=3, eval_every_steps=10,
               model_resume_override=None, phase_name="Attack",
               return_gaussians=False, ref_gaussians=None,
               model_state_dict_override=None,
               *,
               gaussian_export_loader=None,
               gaussian_export_num_samples: int = 32,
               gaussian_export_path: str = None,
               gaussian_export_stage: str = None):
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
        attack_steps: 攻击优化器步数（优先于 attack_epochs）
        num_render: 渲染样本数
        eval_every_steps: 每隔多少优化器步评估一次
        model_resume_override: 覆盖 model.resume（如 "tag:xxx"）
        model_state_dict_override: 可选，直接覆盖模型权重（state_dict，通常用于 defense 不落盘）
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

    # 优先使用 attack_steps，否则使用 attack_epochs
    if attack_steps is None:
        if attack_epochs is None:
            attack_epochs = config['training'].get('attack_epochs')
            if attack_epochs is None:
                attack_epochs = 5
        use_steps = False
    else:
        use_steps = True

    is_semantic_deflection = (supervision_loader is not None)

    print(f"\n{'='*80}")
    print(f"  {phase_name}")
    if use_steps:
        print(f"  Attack optimizer steps: {attack_steps}, eval_every_steps: {eval_every_steps}")
    else:
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
    training_cfg = config.get('training', {}) or {}
    attack_cfg = config.get('attack', {}) or {}
    # Attack fine-tune mode: allow attack.mode override, otherwise inherit training.mode.
    attack_mode = str(attack_cfg.get('mode') or training_cfg.get('mode', 'full') or 'full').lower().strip()
    apply_lora = (attack_mode == 'lora')

    override_loaded = False
    override_keys = list(getattr(model_state_dict_override, "keys", lambda: [])()) if (model_state_dict_override is not None) else []
    override_looks_like_peft = (
        any(str(k).startswith("base_model.") for k in override_keys)
        or any(".lora_" in str(k) for k in override_keys)
        or any(".base_layer." in str(k) for k in override_keys)
    )

    # Important: if the attack model is LoRA-wrapped, load a *vanilla* (non-PEFT) state_dict BEFORE applying LoRA.
    # Otherwise qkv/proj modules are already replaced by PEFT layers (base_layer/lora_A/lora_B) and keys won't match.
    if apply_lora and (model_state_dict_override is not None) and (not override_looks_like_peft):
        model_mgr.setup(apply_lora=False, device='cuda')
        model = model_mgr.model

        raw = model
        while hasattr(raw, 'module'):
            raw = raw.module
        missing, unexpected = raw.load_state_dict(model_state_dict_override, strict=False)
        override_loaded = True

        if missing or unexpected:
            def _summarize_keys(keys, max_prefixes: int = 8):
                from collections import Counter
                if not keys:
                    return ""
                c = Counter()
                for k in keys:
                    prefix = str(k).split('.', 1)[0]
                    c[prefix] += 1
                parts = [f"{p}:{n}" for p, n in c.most_common(max_prefixes)]
                if len(c) > max_prefixes:
                    parts.append(f"...(+{len(c) - max_prefixes} prefixes)")
                return ", ".join(parts)

            lpips_missing = bool(missing) and all(("lpips_loss." in str(k)) for k in missing)
            level = "提示" if (lpips_missing and not unexpected) else "警告"

            print(f"  [run_attack] {level}: state_dict 覆盖非严格匹配 "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
            if missing:
                print(f"    missing prefixes: {_summarize_keys(missing)}")
                print(f"    missing (first 12): {list(missing)[:12]}")
            if unexpected:
                print(f"    unexpected prefixes: {_summarize_keys(unexpected)}")
                print(f"    unexpected (first 12): {list(unexpected)[:12]}")

        # Apply LoRA after loading base weights.
        model_mgr.apply_lora()
        model = model_mgr.model
    else:
        model_mgr.setup(device='cuda')
        model = model_mgr.model

    if (model_state_dict_override is not None) and (not override_loaded):
        # NOTE: When attack is in LoRA mode, `model` is a PEFT wrapper (PeftModel).
        # Defense training produces a *vanilla* LGM state_dict without "base_model.model." prefixes.
        # In that case we must load into the wrapped base model, not the PEFT wrapper.
        raw = model
        while hasattr(raw, 'module'):
            raw = raw.module

        override_keys = list(getattr(model_state_dict_override, "keys", lambda: [])())
        has_peft_prefix = any(str(k).startswith("base_model.") for k in override_keys)

        load_target = raw
        if (not has_peft_prefix) and hasattr(raw, "base_model"):
            base = getattr(raw, "base_model", None)
            if hasattr(base, "model") and hasattr(base.model, "load_state_dict"):
                load_target = base.model
            else:
                # Fallback: PEFT exposes get_base_model() in most versions.
                gbm = getattr(raw, "get_base_model", None)
                if callable(gbm):
                    try:
                        load_target = gbm()
                    except Exception:
                        load_target = raw

        missing, unexpected = load_target.load_state_dict(model_state_dict_override, strict=False)
        if missing or unexpected:
            def _summarize_keys(keys, max_prefixes: int = 8):
                from collections import Counter
                if not keys:
                    return ""
                c = Counter()
                for k in keys:
                    prefix = str(k).split('.', 1)[0]
                    c[prefix] += 1
                parts = [f"{p}:{n}" for p, n in c.most_common(max_prefixes)]
                if len(c) > max_prefixes:
                    parts.append(f"...(+{len(c) - max_prefixes} prefixes)")
                return ", ".join(parts)

            # LGM overrides state_dict() to intentionally drop lpips_loss weights.
            # This commonly leads to a fixed number of missing keys when loading into a fresh model
            # that has lpips_loss instantiated. This is expected and harmless.
            lpips_missing = bool(missing) and all(("lpips_loss." in str(k)) for k in missing)
            level = "提示" if (lpips_missing and not unexpected) else "警告"

            print(f"  [run_attack] {level}: state_dict 覆盖非严格匹配 "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
            if missing:
                print(f"    missing prefixes: {_summarize_keys(missing)}")
                print(f"    missing (first 12): {list(missing)[:12]}")
            if unexpected:
                print(f"    unexpected prefixes: {_summarize_keys(unexpected)}")
                print(f"    unexpected (first 12): {list(unexpected)[:12]}")

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

    # ---------------------------------------------------------------------
    # Optional debug instrumentation (params / trap metrics during attack)
    # ---------------------------------------------------------------------
    debug_cfg = (config.get('attack', {}).get('debug', {}) or {})
    debug_enabled = bool(debug_cfg.get('enabled', False))
    debug_track_params = debug_cfg.get('track_params') or [
        # Head (small, attacker-friendly)
        'conv.weight',
        'conv.bias',
    ]
    debug_track_trap_metrics = bool(debug_cfg.get('trap_metrics', True))
    debug_track_features = bool(debug_cfg.get('feature_stats', True))

    # Optional: capture UNet output features (conv input) to diagnose "head-only shortcut" behavior.
    # This is cheap and helps verify whether a feature-space defense is actually active / recoverable.
    feature_hook_handle = None
    if debug_enabled and debug_track_features:
        try:
            model._debug_conv_in = None

            def _capture_conv_input(module, input, output):
                model._debug_conv_in = input[0]

            feature_hook_handle = model.conv.register_forward_hook(_capture_conv_input)
            print("  [Debug] Feature hook enabled: conv input (UNet output)")
        except Exception as e:
            feature_hook_handle = None
            print(f"  [Debug] Feature hook disabled due to error: {type(e).__name__}: {e}")

    # Resolve parameters to track (keep this small; copying large tensors each eval is expensive).
    tracked_param_refs = {}
    tracked_param_live = {}
    if debug_enabled:
        name_to_param = {n: p for n, p in model.named_parameters()}
        for name in debug_track_params:
            p = name_to_param.get(name)
            resolved_name = name
            if p is None:
                # PEFT (LoRA) wraps the base model under "base_model.model.*".
                cand = f"base_model.model.{name}"
                p = name_to_param.get(cand)
                if p is not None:
                    resolved_name = cand
            if p is None:
                continue
            tracked_param_live[resolved_name] = p
            tracked_param_refs[resolved_name] = p.detach().float().cpu().clone()
        print(f"  [Debug] Attack instrumentation enabled: track_params={list(tracked_param_live.keys())}")

    # Prepare a fixed batch for trap metrics (no rendering; cheap compared to Evaluator source-eval).
    trap_fns = None
    diag_input_images = None
    diag_batch_full = None
    if debug_enabled and debug_track_trap_metrics:
        try:
            from methods.trap_losses import (
                PositionCollapseLoss,
                ScaleAnisotropyLoss,
                ScaleMagnitudeCollapseLoss,
                OpacityCollapseLoss,
                OpacityLogitCollapseLoss,
                OpacityLogitHybridCollapseLoss,
                RotationAnisotropyLoss,
                ColorCollapseLoss,
            )

            defense_trap_cfg = (config.get('defense', {}).get('trap_losses', {}) or {})
            trap_fns = {}
            if defense_trap_cfg.get('position', {}).get('static', False):
                trap_fns['trap_position'] = PositionCollapseLoss().to('cuda')
            if defense_trap_cfg.get('scale', {}).get('static', False):
                scfg = defense_trap_cfg.get('scale', {}) or {}
                smode = str(scfg.get('mode', 'anisotropy')).lower().strip()
                if smode in ('collapse', 'magnitude', 'mag', 'magnitude_collapse', 'log'):
                    trap_fns['trap_scale'] = ScaleMagnitudeCollapseLoss(
                        epsilon=float(scfg.get('epsilon', 1e-8) or 1e-8)
                    ).to('cuda')
                else:
                    trap_fns['trap_scale'] = ScaleAnisotropyLoss().to('cuda')
            if defense_trap_cfg.get('opacity', {}).get('static', False):
                ocfg = defense_trap_cfg.get('opacity', {}) or {}
                mode = str(ocfg.get('mode', 'log')).lower().strip()
                if mode in ('logit_hybrid', 'hybrid_logit', 'hybrid'):
                    trap_fns['trap_opacity'] = OpacityLogitHybridCollapseLoss(
                        epsilon=float(ocfg.get('epsilon', 1e-6) or 1e-6),
                        topk_frac=ocfg.get('topk_frac'),
                        topk_k=ocfg.get('topk_k'),
                        bulk_weight=float(ocfg.get('bulk_weight', 1.0) or 1.0),
                        tail_weight=float(ocfg.get('tail_weight', 1.0) or 1.0),
                    ).to('cuda')
                else:
                    use_logit = bool(ocfg.get('use_logit', False)) or (mode in ('logit', 'logits'))
                    opa_cls = OpacityLogitCollapseLoss if use_logit else OpacityCollapseLoss
                    trap_fns['trap_opacity'] = opa_cls(
                        epsilon=float(ocfg.get('epsilon', 1e-6) or 1e-6),
                        topk_frac=ocfg.get('topk_frac'),
                        topk_k=ocfg.get('topk_k'),
                    ).to('cuda')
            if defense_trap_cfg.get('rotation', {}).get('static', False):
                trap_fns['trap_rotation'] = RotationAnisotropyLoss().to('cuda')
            if defense_trap_cfg.get('color', {}).get('static', False):
                trap_fns['trap_color'] = ColorCollapseLoss().to('cuda')

            # Fallback: if config has no active traps, still track the 4 built-in diagnostics.
            if not trap_fns:
                trap_fns = {
                    'trap_position': PositionCollapseLoss().to('cuda'),
                    'trap_scale': ScaleAnisotropyLoss().to('cuda'),
                    'trap_opacity': OpacityCollapseLoss().to('cuda'),
                    'trap_rotation': RotationAnisotropyLoss().to('cuda'),
                }

            diag_loader = target_eval_loader if (is_semantic_deflection and target_eval_loader is not None) else target_train_loader
            diag_batch = next(iter(diag_loader))
            diag_batch_full = diag_batch
            diag_input_images = diag_batch['input_images'].to('cuda')
            print(f"  [Debug] Trap metrics enabled: keys={list(trap_fns.keys())}, "
                  f"diag_batch_B={diag_input_images.shape[0]}")
        except Exception as e:
            trap_fns = None
            diag_input_images = None
            diag_batch_full = None
            print(f"  [Debug] Trap metrics disabled due to error: {type(e).__name__}: {e}")

    step_history = []
    global_step = 0
    interval_loss, interval_lpips, interval_masked_psnr, interval_masked_lpips = 0, 0, 0, 0
    interval_count = 0

    # ---------------------------------------------------------------------
    # Pre-attack diagnostics (helps verify defense transfer / initial state)
    # ---------------------------------------------------------------------
    pre_eval_cfg = (config.get('attack', {}).get('pre_eval', {}) or {})
    pre_eval_enabled = bool(pre_eval_cfg.get('enabled', False)) or debug_enabled
    pre_eval_render_metrics = bool(pre_eval_cfg.get('render_metrics', False))
    pre_eval_num_samples = pre_eval_cfg.get('num_samples', None)
    if pre_eval_num_samples is not None:
        try:
            pre_eval_num_samples = int(pre_eval_num_samples)
        except Exception:
            pre_eval_num_samples = None

    if pre_eval_enabled:
        pre_diag = {'step': 0, 'epoch': 0}
        try:
            # (1) Fast Gaussian-space stats on the fixed diag batch (no rendering)
            if (diag_input_images is not None) and (trap_fns is not None):
                model.eval()
                with torch.no_grad():
                    ga0 = model.forward_gaussians(diag_input_images).float()
                    opacity0 = ga0[..., 3:4].clamp(min=1e-12, max=1.0)
                    pre_diag['gaussian_opacity_mean'] = float(opacity0.mean().item())
                    pre_diag['gaussian_pos_spread'] = float(ga0[..., 0:3].std(dim=1).mean().item())
                    pre_diag['gaussian_scale_mean'] = float(ga0[..., 4:7].mean().item())
                    try:
                        # Opacity distribution (helps detect "near-zero but still trainable" vs "deep dead-zone").
                        op_flat = opacity0.reshape(-1)
                        pre_diag['gaussian_opacity_min'] = float(op_flat.min().item())
                        # Quantiles are small tensors; OK in no_grad.
                        for q in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99):
                            pre_diag[f'gaussian_opacity_p{int(q*100):02d}'] = float(
                                torch.quantile(op_flat, q).item()
                            )
                        # Sigmoid derivative proxy (gradient attenuation from opacity->raw logit).
                        deriv = opacity0 * (1.0 - opacity0)
                        pre_diag['gaussian_opacity_sigmoid_deriv_mean'] = float(deriv.mean().item())
                        # Logit (≈ raw opacity logit when epsilon is tiny).
                        logit = torch.log(opacity0) - torch.log1p(-opacity0)
                        pre_diag['gaussian_opacity_logit_mean'] = float(logit.mean().item())
                        pre_diag['gaussian_opacity_logit_min'] = float(logit.min().item())
                    except Exception as e:
                        pre_diag['gaussian_opacity_stats_error'] = f"{type(e).__name__}: {e}"
                    for k, fn in trap_fns.items():
                        pre_diag[f'pre_{k}'] = float(fn(ga0).item())

                    # Optional: feature stats (conv input) if hook is enabled.
                    feat0 = getattr(model, "_debug_conv_in", None)
                    if feat0 is not None:
                        f0 = feat0.detach().float()
                        pre_diag['feature_spatial_var'] = float(
                            f0.var(dim=(-2, -1), unbiased=False).mean().item()
                        )
                        pre_diag['feature_mean_abs'] = float(f0.abs().mean().item())

                print(f"  [Pre-Attack] Gaussian stats (fixed batch): "
                      f"opacity_mean={pre_diag['gaussian_opacity_mean']:.4f}, "
                      f"pos_spread={pre_diag['gaussian_pos_spread']:.4f}, "
                      f"scale_mean={pre_diag['gaussian_scale_mean']:.6f}")
                if 'feature_spatial_var' in pre_diag:
                    print(f"  [Pre-Attack] Feature stats (conv input): "
                          f"spatial_var={pre_diag['feature_spatial_var']:.4f}, "
                          f"mean_abs={pre_diag.get('feature_mean_abs', 0.0):.4f}")
                trap_items = [(k, pre_diag.get(f'pre_{k}')) for k in (trap_fns or {}).keys()]
                trap_items = [(k, v) for k, v in trap_items if v is not None]
                if trap_items:
                    print("  [Pre-Attack] Trap metrics (fixed batch): " +
                          ", ".join([f"{k}={v:.4f}" for k, v in trap_items]))

            # (1.5) Optional: gradient norms of the *attack loss* at step 0 (one batch).
            grad_cfg = (debug_cfg.get('grad_stats', {}) or {}) if debug_enabled else {}
            grad_enabled = grad_cfg.get('enabled', None)
            if grad_enabled is None:
                grad_enabled = bool(debug_enabled)
            grad_enabled = bool(grad_enabled)

            if grad_enabled:
                import math
                grad_batch = diag_batch_full
                if grad_batch is None:
                    grad_batch = next(iter(target_train_loader))

                # Compute gradients once (no optimizer step; does not change model weights).
                model.train()
                finetuner.optimizer.zero_grad()

                data = finetuner._prepare_data(grad_batch)

                mp = str(getattr(finetuner, "mixed_precision", "bf16")).lower()
                if mp == 'bf16':
                    with autocast('cuda', dtype=torch.bfloat16):
                        results = model.forward(data, step_ratio=1.0)
                elif mp == 'fp16':
                    with autocast('cuda', dtype=torch.float16):
                        results = model.forward(data, step_ratio=1.0)
                else:
                    results = model.forward(data, step_ratio=1.0)

                loss_attack = results['loss']
                ga_pred = results.get('gaussians')
                if isinstance(ga_pred, torch.Tensor):
                    try:
                        ga_pred.retain_grad()
                    except Exception:
                        ga_pred = None

                loss_scaled = loss_attack / max(int(finetuner.gradient_accumulation_steps), 1)
                loss_scaled.backward()

                # Group gradients by parameter name.
                target_layer_keys = (config.get('defense', {}).get('target_layers') or [])
                target_layer_keys = [str(k) for k in target_layer_keys if k]

                def _sum_grad_sq(match_fn):
                    s = 0.0
                    for name, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if not match_fn(name, p):
                            continue
                        g = p.grad
                        if g is None:
                            continue
                        s += float(g.detach().float().pow(2).sum().item())
                    return s

                total_sq = _sum_grad_sq(lambda _n, _p: True)
                head_sq = _sum_grad_sq(lambda n, _p: n.startswith('conv.'))
                unet_sq = _sum_grad_sq(lambda n, _p: n.startswith('unet.'))
                tl_sq = _sum_grad_sq(lambda n, _p: any(k in n for k in target_layer_keys))

                pre_diag['pre_attack_loss'] = float(loss_attack.detach().float().item())
                pre_diag['grad_total_norm'] = float(math.sqrt(max(total_sq, 0.0)))
                pre_diag['grad_head_norm'] = float(math.sqrt(max(head_sq, 0.0)))
                pre_diag['grad_unet_norm'] = float(math.sqrt(max(unet_sq, 0.0)))
                pre_diag['grad_target_layers_norm'] = float(math.sqrt(max(tl_sq, 0.0)))

                head_frac = (head_sq / total_sq) if total_sq > 0 else 0.0
                tl_frac = (tl_sq / total_sq) if total_sq > 0 else 0.0
                pre_diag['grad_head_frac'] = float(head_frac)
                pre_diag['grad_target_layers_frac'] = float(tl_frac)

                print(f"  [Pre-Attack][Grad] loss={pre_diag['pre_attack_loss']:.4f}, "
                      f"||g||={pre_diag['grad_total_norm']:.2f}, "
                      f"head={pre_diag['grad_head_norm']:.2f} ({head_frac:.1%}), "
                      f"target_layers={pre_diag['grad_target_layers_norm']:.2f} ({tl_frac:.1%})")

                # Extra: gradient on Gaussian outputs (what attributes are easiest to "repair"?)
                try:
                    if isinstance(ga_pred, torch.Tensor) and isinstance(ga_pred.grad, torch.Tensor):
                        g = ga_pred.grad.detach().float()
                        ga = ga_pred.detach().float()

                        def _gn(slice_):
                            return float(g[..., slice_].norm().item())

                        pre_diag['gaussian_grad_norm'] = float(g.norm().item())
                        pre_diag['gaussian_grad_pos_norm'] = _gn(slice(0, 3))
                        pre_diag['gaussian_grad_opacity_norm'] = _gn(slice(3, 4))
                        pre_diag['gaussian_grad_scale_norm'] = _gn(slice(4, 7))
                        pre_diag['gaussian_grad_rotation_norm'] = _gn(slice(7, 11))
                        pre_diag['gaussian_grad_color_norm'] = _gn(slice(11, 14))

                        # Estimate "raw opacity logit" gradient magnitude: dL/draw = dL/do * o*(1-o)
                        o = ga[..., 3:4].clamp(min=1e-12, max=1.0)
                        go = g[..., 3:4]
                        graw = go * (o * (1.0 - o))
                        pre_diag['gaussian_grad_opacity_raw_est_norm'] = float(graw.norm().item())

                        # Coverage of deep-dead-zone (very small opacities)
                        op_flat = o.reshape(-1)
                        pre_diag['gaussian_opacity_lt_5e-3_frac'] = float((op_flat < 5e-3).float().mean().item())
                        pre_diag['gaussian_opacity_lt_1e-3_frac'] = float((op_flat < 1e-3).float().mean().item())
                        pre_diag['gaussian_opacity_lt_1e-4_frac'] = float((op_flat < 1e-4).float().mean().item())
                        pre_diag['gaussian_opacity_sigmoid_deriv_mean_gradbatch'] = float(
                            (o * (1.0 - o)).mean().item()
                        )

                        # Small console hint (keep it short).
                        print(f"  [Pre-Attack][GGrad] ||dL/dg||={pre_diag['gaussian_grad_norm']:.2f}, "
                              f"opacity_raw_est={pre_diag['gaussian_grad_opacity_raw_est_norm']:.2e}, "
                              f"opa<1e-3={pre_diag['gaussian_opacity_lt_1e-3_frac']:.1%}")
                except Exception as e:
                    pre_diag['gaussian_grad_stats_error'] = f"{type(e).__name__}: {e}"

                # Extra: top-k gradient parameters/groups (helps diagnose shortcut channels)
                try:
                    from collections import defaultdict

                    grad_items = []
                    for name, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        g = p.grad
                        if g is None:
                            continue
                        gn = float(g.detach().float().norm().item())
                        if gn <= 0:
                            continue
                        grad_items.append((gn, name, int(p.numel())))

                    grad_items.sort(key=lambda x: x[0], reverse=True)

                    top_params = []
                    for gn, name, numel in grad_items[:10]:
                        top_params.append({'name': name, 'norm': float(gn), 'numel': int(numel)})
                    if top_params:
                        pre_diag['top_grad_params'] = top_params

                    group_sq = defaultdict(float)
                    group_numel = defaultdict(int)
                    for gn, name, numel in grad_items:
                        parts = name.split('.')
                        key = '.'.join(parts[:3]) if len(parts) >= 3 else parts[0]
                        group_sq[key] += float(gn) * float(gn)
                        group_numel[key] += int(numel)
                    group_items = [
                        (math.sqrt(max(sq, 0.0)), key, int(group_numel.get(key, 0)))
                        for key, sq in group_sq.items()
                    ]
                    group_items.sort(key=lambda x: x[0], reverse=True)
                    top_groups = []
                    for gn, key, numel in group_items[:10]:
                        top_groups.append({'name': key, 'norm': float(gn), 'numel': int(numel)})
                    if top_groups:
                        pre_diag['top_grad_groups'] = top_groups
                        pretty = ', '.join([f"{g['name']}:{g['norm']:.2f}" for g in top_groups[:5]])
                        print(f"  [Pre-Attack][GradTop] {pretty}")
                except Exception as e:
                    pre_diag['top_grad_error'] = f"{type(e).__name__}: {e}"

                finetuner.optimizer.zero_grad()

            # (2) Optional: masked PSNR/LPIPS before any optimization (rendering-based)
            if pre_eval_render_metrics:
                pre_loader = target_eval_loader if (is_semantic_deflection and target_eval_loader is not None) else target_train_loader
                pre_target = evaluator.evaluate_on_loader(pre_loader, num_samples=pre_eval_num_samples)
                pre_diag['masked_psnr'] = float(pre_target.get('psnr', 0.0))
                pre_diag['masked_lpips'] = float(pre_target.get('lpips', 0.0))
                n_str = pre_eval_num_samples if pre_eval_num_samples is not None else 'all'
                print(f"  [Pre-Attack] Target (masked) quality: PSNR={pre_diag['masked_psnr']:.2f}, "
                      f"LPIPS={pre_diag['masked_lpips']:.4f} (num_samples={n_str})")

        except Exception as e:
            print(f"  [Pre-Attack] Diagnostics skipped due to error: {type(e).__name__}: {e}")

        # Only record if we actually captured something beyond (step, epoch).
        if len(pre_diag) > 2:
            step_history.append(pre_diag)

    # 计算总优化器步数
    if use_steps:
        total_steps = attack_steps
    else:
        # epoch模式：总优化器步数 = epochs × 每个epoch的batch数 / 梯度累积步数
        batches_per_epoch = len(target_train_loader)
        total_steps = (attack_epochs * batches_per_epoch + finetuner.gradient_accumulation_steps - 1) // finetuner.gradient_accumulation_steps

    # 时间跟踪
    import time
    training_start_time = time.time()
    step_times = []  # 记录最近的步时间用于平滑估计
    last_step_time = time.time()

    # 调试：训练前参数校验
    param_sum_before = sum(p.data.sum().item() for p in model.parameters())
    print(f"  [DEBUG] 训练前参数 sum: {param_sum_before:.6f}")

    epoch = 0
    while True:
        epoch += 1
        model.train()

        # Step 模式：每个 epoch 重新打乱数据（确保不会一直训练前面的样本）
        # Epoch 模式：正常遍历
        train_iterator = iter(target_train_loader)

        for batch_idx in range(len(target_train_loader)):
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Step 模式下可能在 epoch 中途退出，这里不应该到达
                break

            loss_dict, updated = finetuner.train_step(batch)

            # 只在实际更新优化器时计数和记录
            if updated:
                global_step += 1

                # 计算时间统计
                current_time = time.time()
                step_time = current_time - last_step_time
                step_times.append(step_time)
                # 保持最近20步的时间用于平滑估计
                if len(step_times) > 20:
                    step_times.pop(0)
                avg_step_time = sum(step_times) / len(step_times)
                last_step_time = current_time

                interval_loss += loss_dict['loss']
                interval_lpips += loss_dict.get('loss_lpips', 0)
                interval_masked_psnr += loss_dict.get('masked_psnr', 0)
                interval_masked_lpips += loss_dict.get('masked_lpips', 0)
                interval_count += 1

                if global_step % eval_every_steps == 0 or global_step == total_steps:
                    metrics = {
                        'step': global_step,
                        'epoch': epoch,
                        'loss': interval_loss / interval_count if interval_count > 0 else 0,
                        'lpips': interval_lpips / interval_count if interval_count > 0 else 0,
                        'masked_lpips': interval_masked_lpips / interval_count if interval_count > 0 else 0,
                        'masked_psnr': interval_masked_psnr / interval_count if interval_count > 0 else 0,
                    }

                    # Debug: parameter deltas (small, CPU-safe)
                    if debug_enabled and tracked_param_live:
                        for pname, p in tracked_param_live.items():
                            ref = tracked_param_refs.get(pname)
                            if ref is None:
                                continue
                            cur = p.detach().float().cpu()
                            delta = cur - ref
                            key = pname.replace('.', '_')
                            metrics[f'param_{key}_norm'] = float(cur.norm().item())
                            metrics[f'param_{key}_delta_norm'] = float(delta.norm().item())

                    # Debug: Gaussian-space trap metrics on a fixed batch (no rendering)
                    if debug_enabled and (trap_fns is not None) and (diag_input_images is not None):
                        try:
                            with torch.no_grad():
                                g = model.forward_gaussians(diag_input_images).float()
                                metrics['gaussian_opacity_mean'] = float(g[..., 3].mean().item())
                                metrics['gaussian_scale_mean'] = float(g[..., 4:7].mean().item())
                                metrics['gaussian_pos_std'] = float(g[..., 0:3].std().item())
                                for k, fn in trap_fns.items():
                                    metrics[k] = float(fn(g).item())
                                feat = getattr(model, "_debug_conv_in", None)
                                if feat is not None:
                                    f = feat.detach().float()
                                    metrics['feature_spatial_var'] = float(
                                        f.var(dim=(-2, -1), unbiased=False).mean().item()
                                    )
                                    metrics['feature_mean_abs'] = float(f.abs().mean().item())
                        except Exception as e:
                            metrics['debug_trap_error'] = f"{type(e).__name__}: {e}"

                    # 轻量：每次 step-eval 同时评估 source（用于 plot_pipeline_results 的 Source 曲线）
                    try:
                        model.eval()
                        has_lora = hasattr(model, 'disable_adapter_layers')
                        if has_lora:
                            model.disable_adapter_layers()
                        src_eval = evaluator.evaluate_on_loader(source_val_loader)
                        metrics['source_psnr'] = float(src_eval.get('psnr', 0))
                        metrics['source_lpips'] = float(src_eval.get('lpips', 0))
                    except Exception as e:
                        # 评估失败不应中断攻击主流程
                        print(f"  [run_attack] 警告: source-eval 失败 (step={global_step}): {e}")
                        metrics['source_psnr'] = 0.0
                        metrics['source_lpips'] = 0.0
                    finally:
                        if hasattr(model, 'enable_adapter_layers'):
                            try:
                                model.enable_adapter_layers()
                            except Exception:
                                pass
                        model.train()
                    step_history.append(metrics)

                    # 记录效率指标
                    if hasattr(config, '_efficiency_tracker') and config._efficiency_tracker is not None:
                        config._efficiency_tracker.record(
                            step=global_step,
                            epoch=epoch,
                            step_time=avg_step_time,
                            masked_psnr=metrics.get('masked_psnr'),
                            masked_lpips=metrics.get('masked_lpips'),
                            psnr=metrics.get('psnr'),
                            lpips=metrics.get('lpips'),
                            loss=metrics.get('loss'),
                        )

                    # 计算ETA
                    eta_str = ""
                    if len(step_times) >= 3:
                        remaining_steps = total_steps - global_step
                        eta_seconds = remaining_steps * avg_step_time
                        if eta_seconds < 60:
                            eta_str = f", ETA: {int(eta_seconds)}s"
                        elif eta_seconds < 3600:
                            eta_str = f", ETA: {int(eta_seconds/60)}m{int(eta_seconds%60)}s"
                        else:
                            hours = int(eta_seconds / 3600)
                            minutes = int((eta_seconds % 3600) / 60)
                            eta_str = f", ETA: {hours}h{minutes}m"

                    print(f"  [{phase_name}] Optimizer Step {global_step}/{total_steps} (Ep{epoch}) - "
                          f"Loss: {metrics['loss']:.4f}, "
                          f"LPIPS: {metrics['masked_lpips']:.4f}, "
                          f"PSNR: {metrics['masked_psnr']:.2f}, "
                          f"SourcePSNR: {metrics['source_psnr']:.2f}, "
                          f"SourceLPIPS: {metrics['source_lpips']:.4f}{eta_str}")

                    interval_loss, interval_lpips, interval_masked_psnr, interval_masked_lpips = 0, 0, 0, 0
                    interval_count = 0

                # Step 模式：达到目标优化器步数后退出
                if use_steps and global_step >= total_steps:
                    break

        # Step 模式：达到目标优化器步数后退出外层循环
        if use_steps and global_step >= total_steps:
            break

        # Epoch 模式：达到目标 epoch 后退出
        if not use_steps and epoch >= attack_epochs:
            break

        # epoch 结束，flush 残余梯度
        if finetuner._accumulation_counter % finetuner.gradient_accumulation_steps != 0:
            if finetuner.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), finetuner.gradient_clip)
            finetuner.optimizer.step()
            finetuner.optimizer.zero_grad()
            # 这里也算一次优化器更新
            global_step += 1

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

    # Optional: export per-sample gaussians for paper-style distribution plots.
    if gaussian_export_path and gaussian_export_loader is not None:
        try:
            from datetime import datetime
            os.makedirs(os.path.dirname(gaussian_export_path), exist_ok=True)
            stage = gaussian_export_stage or phase_name
            export_obj = evaluator.collect_gaussian_samples(
                gaussian_export_loader,
                num_samples=gaussian_export_num_samples,
                stage=str(stage),
                include_inputs=False,
            )
            export_obj["format_version"] = 1
            export_obj["created_at"] = datetime.now().isoformat()
            export_obj["phase_name"] = str(phase_name)
            export_obj["eval_every_steps"] = int(eval_every_steps)
            export_obj["attack_steps"] = int(attack_steps) if attack_steps is not None else None
            export_obj["attack_epochs"] = int(attack_epochs) if attack_epochs is not None else None
            torch.save(export_obj, gaussian_export_path)
            print(f"  [GaussianExport] saved: {gaussian_export_path} "
                  f"(n={export_obj.get('num_collected', 0)})")
        except Exception as e:
            print(f"  [GaussianExport] skipped due to error: {type(e).__name__}: {e}")

    # Cleanup debug hooks (important when running multiple phases in a single process).
    if feature_hook_handle is not None:
        try:
            feature_hook_handle.remove()
        except Exception:
            pass
        feature_hook_handle = None

    del finetuner, evaluator, model, model_mgr
    torch.cuda.empty_cache()

    if return_gaussians:
        return step_history, source_metrics, target_metrics, gaussians_list
    return step_history, source_metrics, target_metrics
