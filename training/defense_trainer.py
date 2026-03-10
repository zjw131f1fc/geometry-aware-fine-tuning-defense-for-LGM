"""
防御训练器 - 支持 GeoTrap 和 Naive Unlearning 两种防御方法

defense.method 配置：
- geotrap: 几何陷阱防御（Trap Loss + 两两乘法耦合 + 特征空间梯度冲突 + 参数加噪双前向）
- naive_unlearning: 朴素遗忘（对 target 渲染 loss 做梯度上升 + source 蒸馏）
- none: 跳过防御（在 load_or_train_defense 中直接返回）

核心机制：
1. Source Data → Distillation Loss（保持原有能力）
2. Target Data → Trap Loss（制造几何陷阱）
   - 干净权重前向 → trap loss（直接优化）
   - 加噪权重前向 → trap loss（鲁棒性，确保邻域内有效）
3. 两两乘法耦合：∑_{i<j} -((1-L_i)(1-L_j)-1) / C(n,2)
4. 特征空间梯度冲突：在 conv 层输入（U-Net 共享特征瓶颈）上计算不同 trap 梯度的
   cosine similarity，最小化之（推向 -1），使攻击者修复一个 trap 时加深另一个
5. 敏感层选择性微调（可选）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import os
import hashlib
import json

if TYPE_CHECKING:
    # Avoid importing heavy dependencies at module import time.
    # These are imported lazily inside methods when needed.
    from models import ModelManager  # noqa: F401
    from data import DataManager  # noqa: F401


class CachedGaussianDataset(Dataset):
    """包装数据集，附带预计算的教师模型 Gaussian 输出

    使用共享内存优化，支持 num_workers > 0 和 batch_size > 1 同时使用。
    """

    def __init__(self, base_dataset, cached_gaussians: List[torch.Tensor]):
        self.base_dataset = base_dataset
        # 将 List[Tensor] stack 成单个 Tensor 并移到共享内存
        # 这样可以避免多进程 DataLoader 时的序列化开销和内存复制
        self.cached_gaussians = torch.stack(cached_gaussians).share_memory_()

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

        # Defense mixed precision (independent from attack). Default: follow training.mixed_precision.
        mp = self.defense_config.get('mixed_precision')
        if mp is None:
            mp = config.get('training', {}).get('mixed_precision', 'bf16')
        if isinstance(mp, bool):
            mp = 'fp16' if mp else 'no'
        self.defense_mixed_precision = str(mp).lower()

        # 陷阱损失（仅 geotrap 需要）
        self.trap_losses = {}
        if self.method == 'geotrap':
            self._setup_trap_losses()

        # 参数加噪配置（仅 geotrap 需要）
        if self.method == 'geotrap':
            robust_config = self.defense_config.get('robustness', {})
            self.use_param_noise = robust_config.get('enabled', False)
            self.noise_scale_target = robust_config.get('noise_scale', 0.01)
            self.noise_warmup_steps = robust_config.get('warmup_steps', 0)
            self.current_noise_scale = 0.0 if self.noise_warmup_steps > 0 else self.noise_scale_target
        else:
            self.use_param_noise = False
            self.noise_scale_target = 0
            self.noise_warmup_steps = 0
            self.current_noise_scale = 0

        # 输入加噪配置（仅 geotrap 需要）
        if self.method == 'geotrap':
            input_noise_config = self.defense_config.get('input_noise', {})
            self.use_input_noise = input_noise_config.get('enabled', False)
            self.input_noise_scale_target = input_noise_config.get('noise_scale', 0.02)
            self.input_noise_warmup_steps = input_noise_config.get('warmup_steps', 0)
            self.current_input_noise_scale = 0.0 if self.input_noise_warmup_steps > 0 else self.input_noise_scale_target
        else:
            self.use_input_noise = False
            self.input_noise_scale_target = 0
            self.input_noise_warmup_steps = 0
            self.current_input_noise_scale = 0

        # 特征空间梯度冲突配置（仅 geotrap 且 trap >= 2 时有效）
        if self.method == 'geotrap':
            conflict_config = self.defense_config.get('gradient_conflict', {})
            self.use_gradient_conflict = conflict_config.get('enabled', False)
            self.conflict_weight = conflict_config.get('weight', 1.0)
            self.conflict_every_k = conflict_config.get('every_k_steps', 1)
        else:
            self.use_gradient_conflict = False

        # Feature-space trap（可选，一阶，免渲染）：
        # 目标：让 target 输入在 UNet→head(conv) 的共享瓶颈特征上发生“空间塌缩”，
        # 从而削弱攻击者仅微调 head(conv) 的快速修复通道（conv 是 1×1，无法凭空生成空间结构）。
        # 注意：该项依赖于 conv 输入 forward hook（与 gradient_conflict 共用）。
        if self.method == 'geotrap':
            ft_cfg = self.defense_config.get('feature_trap', {}) or {}
            self.use_feature_trap = bool(ft_cfg.get('enabled', False))
            self.feature_trap_weight = float(
                ft_cfg.get('weight', ft_cfg.get('lambda', ft_cfg.get('lambda_feature', 0.0) or 0.0))
            )
            # Enabled but weight missing: default to 1.0 to avoid "enabled but no effect".
            if self.use_feature_trap and self.feature_trap_weight == 0.0:
                self.feature_trap_weight = 1.0
            self.feature_trap_mode = str(ft_cfg.get('mode', 'spatial_var')).lower().strip()

            # Optional: restrict channels, e.g., [0,1,2,3,4,5,6] for geometry-related channels.
            ch = ft_cfg.get('channels', None)
            if ch is None or ch == 'all':
                self.feature_trap_channels = None
            else:
                if isinstance(ch, (list, tuple)):
                    parsed = []
                    for v in ch:
                        try:
                            parsed.append(int(v))
                        except Exception:
                            continue
                    self.feature_trap_channels = parsed if parsed else None
                else:
                    try:
                        self.feature_trap_channels = [int(ch)]
                    except Exception:
                        self.feature_trap_channels = None
        else:
            self.use_feature_trap = False
            self.feature_trap_weight = 0.0
            self.feature_trap_mode = 'spatial_var'
            self.feature_trap_channels = None

        # 梯度累积配置（defense 可独立设置；默认继承 training.gradient_accumulation_steps）
        self.gradient_accumulation_steps = self.defense_config.get(
            'gradient_accumulation_steps',
            config.get('training', {}).get('gradient_accumulation_steps', 1),
        )

        # 防御阶段梯度处理（仅影响 defense，不影响攻击/微调评估）
        # 目的：缓解“梯度 shortcut”（少量参数快速达成 trap，后续很快被修复）
        if self.method == 'geotrap':
            gc_cfg = self.defense_config.get('grad_clip', {}) or {}
            self.grad_clip_mode = str(gc_cfg.get('mode', 'norm')).lower()
            # Hard safety clip (global norm). Default: inherit legacy training.gradient_clip.
            self.grad_norm_clip = float(
                gc_cfg.get('norm', config.get('training', {}).get('gradient_clip', 0.0) or 0.0)
            )
            # Energy cap (per-tensor energy cap relative to avg tensor energy).
            self.grad_energy_cap_mult = float(gc_cfg.get('energy_mult', 0.0))
            # Apply-on semantics: energy cap can be restricted to target-only steps
            # to avoid distillation degradation.
            self.grad_energy_apply_on = str(
                gc_cfg.get('energy_apply_on', gc_cfg.get('apply_on', 'all'))
            ).lower()
        else:
            self.grad_clip_mode = 'norm'
            self.grad_norm_clip = float(config.get('training', {}).get('gradient_clip', 0.0) or 0.0)
            self.grad_energy_cap_mult = 0.0
            self.grad_energy_apply_on = 'all'

        # 乘法耦合配置（仅 geotrap 需要）
        if self.method == 'geotrap':
            coupling_config = self.defense_config.get('coupling', {})
            self.coupling_temperature = coupling_config.get('temperature', 0.1)
            self.coupling_use_log = coupling_config.get('use_log_transform', False)
        else:
            self.coupling_temperature = 0.1
            self.coupling_use_log = False

        # Trap 聚合方式（仅 geotrap 需要）
        # - mean: 当前默认，稳定且不会随启用 trap 数量直接放大总 loss
        # - sum: 直接加和（稳定，但会随 trap 数量线性放大）
        # - bottleneck_logsumexp: 近似 max 的瓶颈优化，持续追着“最弱 trap”打
        # - pairwise_multiplicative: 旧版两两乘法耦合（保留兼容）
        if self.method == 'geotrap':
            agg_cfg = self.defense_config.get('trap_aggregation', {}) or {}
            self.trap_aggregation_method = str(
                agg_cfg.get('method', 'mean')
            ).lower()
            self.trap_bottleneck_tau = float(agg_cfg.get('tau', 0.25))
        else:
            self.trap_aggregation_method = 'sum'
            self.trap_bottleneck_tau = 0.25

        # 可选：对每个 trap loss 加权（key 使用 trap_losses dict 的名字，如 'opacity_static'）
        self.trap_weights = (self.defense_config.get('trap_weights', {}) or {}) if self.method == 'geotrap' else {}

        # Anti-shortcut：避免 trap 主要由 head(conv) 的少量参数完成，导致可被快速修复
        # 支持 geotrap 和 naive_unlearning 两种方法
        if self.method in ('geotrap', 'naive_unlearning'):
            anti_cfg = self.defense_config.get('antishortcut', {}) or {}
            self.freeze_head = bool(anti_cfg.get('freeze_head', False))
            self.freeze_head_bias = bool(anti_cfg.get('freeze_head_bias', False))
            # freeze_lora_targets: 冻结 LoRA 可触及的层（qkv/proj），迫使 trap 写入上游 conv 层
            self.freeze_lora_targets = bool(anti_cfg.get('freeze_lora_targets', False))
            self.lora_target_modules = list(anti_cfg.get('lora_target_modules', ['qkv', 'proj']))
            self.head_lr_mult = float(anti_cfg.get('head_lr_mult', 1.0))
            self.head_bias_lr_mult = float(anti_cfg.get('head_bias_lr_mult', self.head_lr_mult))
            self.bias_lr_mult = float(anti_cfg.get('bias_lr_mult', 1.0))

            # Head-attack simulation (first-order):
            # Simulate a small attacker fine-tune step on head(conv) parameters (direction = -∇ render_loss),
            # then enforce trap losses at the perturbed head weights.
            #
            # This directly targets the observed "head-only recovery" shortcut: attackers recover quality with
            # tiny conv.* updates in <~60 steps.
            #
            # Config (recommended location): defense.antishortcut.head_attack_sim
            # - enabled: bool
            # - every_k_steps: int
            # - rho: float (normalized step size; like AWP radius)
            # - lr: float (optional, SGD-like step size; used if rho not set)
            # - weight: float (multiplier on this robust trap term)
            # - norm: 'tensor' / 'global'
            # - exclude_bias: bool
            # - mixed_precision: 'no'/'bf16'/'fp16' (for render_loss forward/backward)
            headsim_cfg = anti_cfg.get('head_attack_sim', anti_cfg.get('headsim', {})) or {}
            self.use_head_attack_sim = bool(headsim_cfg.get('enabled', False))
            self.head_attack_sim_every_k = int(headsim_cfg.get('every_k_steps', headsim_cfg.get('every_k', 1) or 1))
            self.head_attack_sim_rho = headsim_cfg.get('rho', None)
            try:
                self.head_attack_sim_rho = float(self.head_attack_sim_rho) if self.head_attack_sim_rho is not None else None
            except Exception:
                self.head_attack_sim_rho = None
            self.head_attack_sim_lr = headsim_cfg.get('lr', headsim_cfg.get('inner_lr', None))
            try:
                self.head_attack_sim_lr = float(self.head_attack_sim_lr) if self.head_attack_sim_lr is not None else None
            except Exception:
                self.head_attack_sim_lr = None
            self.head_attack_sim_weight = float(headsim_cfg.get('weight', 1.0))
            self.head_attack_sim_norm = str(headsim_cfg.get('norm', 'tensor')).lower().strip()
            self.head_attack_sim_exclude_bias = bool(headsim_cfg.get('exclude_bias', False))
            self.head_attack_sim_mixed_precision = str(
                headsim_cfg.get('mixed_precision', self.config.get('training', {}).get('mixed_precision', 'bf16'))
            ).lower().strip()
            self.head_attack_sim_include_input_supervision = bool(
                headsim_cfg.get('include_input_supervision', True)
            )
            # Optional: render-loss unlearning at the perturbed head weights (gradient ascent on render_loss).
            # This aligns the robustness objective with the actual fine-tuning attacker objective.
            self.head_attack_sim_render_unlearn_weight = float(
                headsim_cfg.get('render_unlearn_weight', headsim_cfg.get('render_unlearn', 0.0) or 0.0)
            )
        else:
            self.freeze_head = False
            self.freeze_head_bias = False
            self.freeze_lora_targets = False
            self.lora_target_modules = ['qkv', 'proj']
            self.head_lr_mult = 1.0
            self.head_bias_lr_mult = 1.0
            self.bias_lr_mult = 1.0
            self.use_head_attack_sim = False
            self.head_attack_sim_every_k = 1
            self.head_attack_sim_rho = None
            self.head_attack_sim_lr = None
            self.head_attack_sim_weight = 0.0
            self.head_attack_sim_norm = 'tensor'
            self.head_attack_sim_exclude_bias = False
            self.head_attack_sim_mixed_precision = self.defense_mixed_precision
            self.head_attack_sim_include_input_supervision = True
            self.head_attack_sim_render_unlearn_weight = 0.0

        # AWP（Adversarial Weight Perturbation）：一阶 min-max，增强 trap 对“有方向的微调修复”鲁棒性
        # 注意：这是额外一次 autograd.grad + 额外一次前向（不需要二阶）。
        if self.method == 'geotrap':
            awp_cfg = self.defense_config.get('awp', {}) or {}
            self.use_awp = bool(awp_cfg.get('enabled', False))
            self.awp_rho = float(awp_cfg.get('rho', 1e-3))
            self.awp_weight = float(awp_cfg.get('weight', 1.0))
            self.awp_every_k = int(awp_cfg.get('every_k_steps', 1))
            self.awp_param_scope = str(awp_cfg.get('param_scope', 'head')).lower()
            self.awp_exclude_bias = bool(awp_cfg.get('exclude_bias', True))
            # Whether to include parameters that are currently frozen (requires_grad=False).
            # Default: only include frozen params for head-scope, because attackers can fine-tune them.
            if 'include_frozen' in awp_cfg:
                self.awp_include_frozen = bool(awp_cfg.get('include_frozen'))
            else:
                self.awp_include_frozen = (self.awp_param_scope in ('head', 'conv', 'gaussian_head'))
            # How to normalize the perturbation:
            # - 'tensor': per-tensor normalize (legacy behavior; safe only for a few tensors like head)
            # - 'global': normalize by global grad norm (recommended when selecting many tensors)
            self.awp_norm = str(awp_cfg.get('norm', '') or awp_cfg.get('norm_mode', '') or '').lower().strip()
            if not self.awp_norm:
                self.awp_norm = 'tensor' if self.awp_param_scope in ('head', 'conv', 'gaussian_head') else 'global'
        else:
            self.use_awp = False
            self.awp_rho = 0.0
            self.awp_weight = 0.0
            self.awp_every_k = 1
            self.awp_param_scope = 'head'
            self.awp_exclude_bias = True
            self.awp_include_frozen = False
            self.awp_norm = 'tensor'

        # Optional: stage-wise anchoring (first-order) to prevent "trap wash-out" during retention recovery.
        #
        # Typical use:
        # - Stage A: plant a deep trap (anchor_weight=0).
        # - Stage B: snapshot weights at Stage A end, then apply an L2 penalty to keep selected
        #           parameters close to that snapshot while distilling on source.
        #
        # This is intentionally simple (no Fisher/EWC), but is often enough to keep tail opacities suppressed.
        anchor_cfg = (self.defense_config.get('anchor') or self.defense_config.get('anchoring') or {}) if self.method == 'geotrap' else {}
        self.anchor_enabled = bool(anchor_cfg.get('enabled', False))
        self.anchor_weight = float(anchor_cfg.get('weight', 0.0) or 0.0)
        self.anchor_apply_on = str(anchor_cfg.get('apply_on', 'both')).lower().strip()
        pats = anchor_cfg.get('patterns', anchor_cfg.get('match', None))
        if pats is None:
            pats = []
        if isinstance(pats, str):
            pats = [p.strip() for p in pats.split(',') if p.strip()]
        if isinstance(pats, (list, tuple)):
            self.anchor_patterns = [str(p) for p in pats if str(p)]
        else:
            self.anchor_patterns = []
        self.anchor_state = None  # {name: Tensor} snapshot on the same device as params

        # 训练状态
        self.target_layers = []
        self.frozen_params = []
        self._train_step_counter = 0  # 用于 gradient conflict 的 every_k_steps

        # Optional: gradient surgery (first-order) to reduce retention damage.
        # Idea: on target/trap steps, project trap gradients to be non-conflicting with
        # source distillation gradients (PCGrad-style).
        gs_cfg = (self.defense_config.get('grad_surgery', {}) or {}) if self.method == 'geotrap' else {}
        self.use_grad_surgery = bool(gs_cfg.get('enabled', False))
        self.grad_surgery_every_k = int(gs_cfg.get('every_k_steps', 1))
        self.grad_surgery_eps = float(gs_cfg.get('eps', 1e-12))
        self.grad_surgery_mode = str(gs_cfg.get('mode', 'pcgrad')).lower().strip()

    def snapshot_anchor_state(self, patterns: List[str] | None = None) -> int:
        """
        Snapshot current parameter values for anchoring.

        Returns:
            n: number of tensors snapshotted
        """
        if not self.anchor_enabled:
            self.anchor_state = None
            return 0
        if self.model_mgr is None or self.model_mgr.model is None:
            raise RuntimeError("anchor snapshot requires model to be initialized (call setup first).")

        pats = patterns if patterns is not None else (self.anchor_patterns or [])
        if not pats:
            # Avoid accidentally duplicating the whole model.
            self.anchor_state = None
            return 0

        model = self.model_mgr.model
        state = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not any(k in name for k in pats):
                continue
            state[name] = p.detach().clone()
        self.anchor_state = state if state else None
        return len(state)

    def _compute_anchor_loss(self, model: torch.nn.Module, is_target_data: bool) -> torch.Tensor | None:
        if (not self.anchor_enabled) or (self.anchor_weight <= 0) or (self.anchor_state is None):
            return None
        apply_on = getattr(self, 'anchor_apply_on', 'both')
        if apply_on == 'source' and is_target_data:
            return None
        if apply_on == 'target' and (not is_target_data):
            return None
        if apply_on not in ('both', 'all', 'source', 'target'):
            # unknown => default to both
            pass

        loss = None
        count = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            ref = self.anchor_state.get(name)
            if ref is None:
                continue
            # Keep ref on the same device/dtype as p.
            if ref.device != p.device or ref.dtype != p.dtype:
                ref = ref.to(device=p.device, dtype=p.dtype)
            term = (p - ref).pow(2).mean()
            loss = term if loss is None else (loss + term)
            count += 1
        if loss is None:
            return None
        if count > 1:
            loss = loss / float(count)
        return loss

    def _autocast_defense(self):
        """
        Defense-stage autocast context.

        Supported:
          - 'no' / 'fp32' / 'none': disable autocast
          - 'bf16' / 'bfloat16': autocast bf16
          - 'fp16' / 'float16': autocast fp16
        """
        mp = getattr(self, 'defense_mixed_precision', 'bf16')
        if mp in ('no', 'none', 'fp32'):
            return torch.autocast('cuda', enabled=False)
        if mp in ('bf16', 'bfloat16'):
            return torch.autocast('cuda', dtype=torch.bfloat16)
        if mp in ('fp16', 'float16'):
            return torch.autocast('cuda', dtype=torch.float16)
        raise ValueError(f"defense.mixed_precision 不支持: {mp}（支持 no/bf16/fp16）")

    def _autocast_head_attack_sim(self):
        """
        Autocast context for head-attack simulation forward/backward (render_loss).

        This is intentionally decoupled from defense.mixed_precision, because:
        - defense trap loss often benefits from fp32 stability (esp. opacity/logit),
        - while render_loss backprop is heavier and can be safely run in bf16 for speed.
        """
        mp = str(getattr(self, 'head_attack_sim_mixed_precision', 'bf16')).lower()
        if mp in ('no', 'none', 'fp32'):
            return torch.autocast('cuda', enabled=False)
        if mp in ('bf16', 'bfloat16'):
            return torch.autocast('cuda', dtype=torch.bfloat16)
        if mp in ('fp16', 'float16'):
            return torch.autocast('cuda', dtype=torch.float16)
        raise ValueError(f"defense.antishortcut.head_attack_sim.mixed_precision 不支持: {mp}（支持 no/bf16/fp16）")

    def _select_head_params(self, model, include_frozen: bool = True, exclude_bias: bool = False):
        selected = []
        for name, param in model.named_parameters():
            if not name.startswith('conv.'):
                continue
            if exclude_bias and name.endswith('.bias'):
                continue
            if (not include_frozen) and (not param.requires_grad):
                continue
            selected.append((name, param))
        return selected

    def _compute_head_attack_sim_trap_loss(self, model, batch, input_images):
        """
        Head-attack simulation (first-order).

        1) Compute one attacker gradient step direction on head(conv) by differentiating render_loss
           (same objective used in the real fine-tuning attack).
        2) Apply a small normalized update (rho) or SGD-like update (lr) to head params.
        3) Compute trap loss at the perturbed head, and add it to defense objective.

        Note: this intentionally does NOT backprop through the update (no second-order).
        """
        if not bool(getattr(self, 'use_head_attack_sim', False)):
            return {}, torch.tensor(0.0, device=input_images.device)

        every_k = int(getattr(self, 'head_attack_sim_every_k', 1) or 1)
        if every_k <= 0:
            every_k = 1
        if getattr(self, '_train_step_counter', 0) % every_k != 0:
            return {}, torch.tensor(0.0, device=input_images.device)

        rho = getattr(self, 'head_attack_sim_rho', None)
        lr = getattr(self, 'head_attack_sim_lr', None)
        if rho is None and lr is None:
            # enabled but no step size -> no-op
            return {'headsim_missing_step_size': 1.0}, torch.tensor(0.0, device=input_images.device)

        exclude_bias = bool(getattr(self, 'head_attack_sim_exclude_bias', False))
        head_params = self._select_head_params(model, include_frozen=True, exclude_bias=exclude_bias)
        if not head_params:
            return {'headsim_no_params': 1.0}, torch.tensor(0.0, device=input_images.device)

        from tools.utils import prepare_lgm_data

        # Temporarily enable grads for head params to obtain attack direction.
        req_backup = {name: p.requires_grad for name, p in head_params}
        try:
            for _, p in head_params:
                if not p.requires_grad:
                    p.requires_grad_(True)

            data = prepare_lgm_data(
                batch, model, self.device,
                include_input_supervision=bool(getattr(self, 'head_attack_sim_include_input_supervision', True)),
            )
            with torch.set_grad_enabled(True):
                with self._autocast_head_attack_sim():
                    results = model.forward(data, step_ratio=1.0)
                attack_loss = results['loss']

            params_only = [p for _, p in head_params]
            grads = torch.autograd.grad(
                outputs=attack_loss,
                inputs=params_only,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        finally:
            for name, p in head_params:
                p.requires_grad_(req_backup.get(name, p.requires_grad))

        # Apply head update (simulate attacker): theta <- theta - step * grad
        backup = {}
        eps = 1e-12
        norm_mode = str(getattr(self, 'head_attack_sim_norm', 'tensor')).lower().strip()

        # Compute global grad norm if requested
        global_norm = None
        if rho is not None and norm_mode in ('global', 'all', 'total'):
            total = None
            for g in grads:
                if g is None:
                    continue
                g2 = g.detach().float().pow(2).sum()
                total = g2 if total is None else (total + g2)
            if total is None:
                total = torch.tensor(0.0, device=input_images.device)
            global_norm = torch.sqrt(total + eps).clamp_min(eps)

        # In-place perturbation on .data (first-order / no version counter)
        for (name, param), grad in zip(head_params, grads):
            if grad is None:
                continue
            g = grad.detach()
            backup[name] = param.data.clone()

            if rho is not None:
                if global_norm is not None:
                    denom = global_norm
                else:
                    denom = g.float().norm().clamp_min(eps)
                delta = (-float(rho) * g / denom).to(dtype=param.data.dtype)
                param.data.add_(delta)
            else:
                # SGD-like step (un-normalized)
                param.data.add_((-float(lr)) * g.to(dtype=param.data.dtype))

        # Forward at perturbed head weights: compute trap losses (Gaussian-space, cheap)
        with torch.set_grad_enabled(True):
            with self._autocast_defense():
                gaussians_headsim = model.forward_gaussians(input_images)
            gaussians_headsim = gaussians_headsim.float()

        headsim_trap_dict, headsim_trap_loss = self.compute_trap_loss(
            gaussians_headsim, model
        )

        # Optional: render-loss unlearning at the perturbed head weights (gradient ascent).
        # We do the backward *here* before restoring head params, to keep gradients consistent with
        # the perturbed head weights (still first-order; no backprop through the perturbation).
        render_unlearn_w = float(getattr(self, 'head_attack_sim_render_unlearn_weight', 0.0) or 0.0)
        render_loss_after = None
        if render_unlearn_w > 0:
            with torch.set_grad_enabled(True):
                with self._autocast_head_attack_sim():
                    results_after = model.forward(data, step_ratio=1.0)
                render_loss_after = results_after['loss']

            # Gradient ascent on render_loss => minimize (-render_loss).
            render_unlearn = (-render_unlearn_w) * render_loss_after
            scaled = render_unlearn / max(int(self.gradient_accumulation_steps), 1)
            scaled.backward()

        # Restore head params
        for name, param in head_params:
            if name in backup:
                param.data.copy_(backup[name])

        info = {
            'headsim_attack_loss': float(attack_loss.detach().float().item()),
            'headsim_rho': float(rho) if rho is not None else 0.0,
            'headsim_lr': float(lr) if lr is not None else 0.0,
        }
        if render_loss_after is not None:
            info['headsim_render_loss_after'] = float(render_loss_after.detach().float().item())
            info['headsim_render_unlearn_weight'] = float(render_unlearn_w)
        # Prefix trap dict to avoid clobbering clean trap metrics
        for k, v in headsim_trap_dict.items():
            info[f'headsim_{k}'] = v

        return info, headsim_trap_loss

    def _setup_trap_losses(self):
        """根据配置创建陷阱损失函数"""
        from methods.trap_losses import (
            ScaleAnisotropyLoss,
            ScaleMagnitudeCollapseLoss,
            PositionCollapseLoss,
            OpacityCollapseLoss,
            OpacityLogitCollapseLoss,
            OpacityLogitHybridCollapseLoss,
            RotationAnisotropyLoss,
            ColorCollapseLoss,
        )

        trap_config = self.defense_config.get('trap_losses', {})

        # Position 陷阱
        if trap_config.get('position', {}).get('static', False):
            self.trap_losses['position_static'] = PositionCollapseLoss()

        # Scale 陷阱
        if trap_config.get('scale', {}).get('static', False):
            sc_cfg = trap_config.get('scale', {}) or {}
            sc_mode = str(sc_cfg.get('mode', 'anisotropy')).lower().strip()
            if sc_mode in ('collapse', 'magnitude', 'mag', 'magnitude_collapse', 'log'):
                self.trap_losses['scale_static'] = ScaleMagnitudeCollapseLoss(
                    epsilon=float(sc_cfg.get('epsilon', 1e-8) or 1e-8)
                )
            else:
                self.trap_losses['scale_static'] = ScaleAnisotropyLoss()

        # Opacity 陷阱
        if trap_config.get('opacity', {}).get('static', False):
            opa_cfg = trap_config.get('opacity', {}) or {}
            # Optional: tail mode to avoid sparse escape (penalize only top-k opacity Gaussians).
            topk_frac = opa_cfg.get('topk_frac')
            topk_k = opa_cfg.get('topk_k')
            mode = str(opa_cfg.get('mode', 'log')).lower().strip()
            if mode in ('logit_hybrid', 'hybrid_logit', 'hybrid'):
                self.trap_losses['opacity_static'] = OpacityLogitHybridCollapseLoss(
                    epsilon=float(opa_cfg.get('epsilon', 1e-6) or 1e-6),
                    topk_frac=topk_frac,
                    topk_k=topk_k,
                    bulk_weight=float(opa_cfg.get('bulk_weight', 1.0) or 1.0),
                    tail_weight=float(opa_cfg.get('tail_weight', 1.0) or 1.0),
                )
            else:
                use_logit = bool(opa_cfg.get('use_logit', False)) or (mode in ('logit', 'logits'))
                opa_cls = OpacityLogitCollapseLoss if use_logit else OpacityCollapseLoss
                self.trap_losses['opacity_static'] = opa_cls(
                    epsilon=float(opa_cfg.get('epsilon', 1e-6) or 1e-6),
                    topk_frac=topk_frac,
                    topk_k=topk_k,
                )

        # Rotation 陷阱
        if trap_config.get('rotation', {}).get('static', False):
            self.trap_losses['rotation_static'] = RotationAnisotropyLoss()

        # Color 陷阱
        if trap_config.get('color', {}).get('static', False):
            self.trap_losses['color_static'] = ColorCollapseLoss()

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
        from models import ModelManager
        self.model_mgr = ModelManager(self.config)
        self.model_mgr.setup(apply_lora=False, device=self.device)
        print(f"  ✓ 学生模型已加载")

        # 注册 conv 层 forward hook（用于特征空间梯度冲突 / feature trap）
        self._conflict_hook_handle = None
        need_hook = bool(getattr(self, "use_feature_trap", False)) or (
            self.use_gradient_conflict and len(self.trap_losses) >= 2
        )
        if need_hook:
            model = self.model_mgr.model
            model._conflict_features = None

            def _capture_conv_input(module, input, output):
                """捕获 conv 层输入（U-Net 共享特征），保留计算图"""
                model._conflict_features = input[0]

            self._conflict_hook_handle = model.conv.register_forward_hook(_capture_conv_input)
            if self.use_gradient_conflict and len(self.trap_losses) >= 2:
                print(f"  ✓ 梯度冲突: 已注册 conv 层 hook (weight={self.conflict_weight})")
            if getattr(self, "use_feature_trap", False) and getattr(self, "feature_trap_weight", 0.0) > 0:
                print(f"  ✓ Feature trap: 已注册 conv 层 hook "
                      f"(mode={getattr(self, 'feature_trap_mode', 'spatial_var')}, "
                      f"weight={getattr(self, 'feature_trap_weight', 0.0)})")

        # 2. 设置敏感层微调
        if target_layers is not None:
            print(f"\n[2/5] 设置敏感层微调...")
            self._setup_selective_finetuning(target_layers)
        else:
            print(f"\n[2/5] 跳过敏感层设置（微调所有层）")

        # 2.1 Anti-shortcut：限制 head(conv) 的快捷解（可选）
        # 注意：freeze_head 只对 geotrap 生效，不对 naive_unlearning 生效
        if self.method == 'geotrap' and (self.freeze_head or self.freeze_head_bias):
            self._apply_antishortcut_freeze(self.model_mgr.model)

        # 2.2 Anti-shortcut：冻结 LoRA 目标层（qkv/proj），迫使 trap 写入上游 conv 层
        if self.method in ('geotrap', 'naive_unlearning') and self.freeze_lora_targets:
            self._apply_lora_target_freeze(self.model_mgr.model)

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

        # Defense 可独立设置 batch size（不影响攻击阶段 training.batch_size）
        defense_batch_size = self.defense_config.get('batch_size')
        if defense_batch_size is None:
            defense_batch_size = self.config['training'].get('batch_size', 1)
        if defense_batch_size <= 0:
            raise ValueError(f"defense.batch_size 必须为正整数，当前: {defense_batch_size}")
        self.defense_batch_size = defense_batch_size
        print(f"  ✓ Defense batch_size: {self.defense_batch_size}")

        # DataManager 读取的是 config.training.batch_size，因此这里用一个局部 config 覆盖 batch_size
        import copy
        defense_data_config = copy.deepcopy(self.config)
        defense_data_config.setdefault('training', {})
        defense_data_config['training']['batch_size'] = self.defense_batch_size

        # Source数据加载器（蒸馏用）
        from data import DataManager
        source_data_mgr = DataManager(defense_data_config, self.model_mgr.opt)
        source_data_mgr.setup_dataloaders(train=True, val=False, subset='source')
        source_full_dataset = source_data_mgr.train_loader.dataset

        # Target数据加载器（defense_target：通过 object_split 自动选择 defense 物体）
        target_data_mgr = DataManager(defense_data_config, self.model_mgr.opt)
        target_data_mgr.setup_dataloaders(train=True, val=True, subset='defense_target')
        self.target_loader = target_data_mgr.train_loader
        self.target_val_loader = target_data_mgr.val_loader

        # 混合比例
        self.source_ratio = data_config.get('source_ratio', 0.5)

        print(f"  ✓ Source数据: {len(source_full_dataset)} 样本")
        print(f"  ✓ Target数据: {len(self.target_loader.dataset)} 样本")
        print(f"  ✓ 混合比例: Source {self.source_ratio:.0%} / Target {1-self.source_ratio:.0%}")

        # 4. 预计算教师模型 Gaussian，替换 source_loader
        #    缓存命中时跳过教师模型加载，直接从磁盘读取
        print("\n[4/5] 准备蒸馏目标（Teacher Gaussians）...")
        # 预计算时用临时 loader 遍历全量 source 数据
        self.source_loader = source_data_mgr.train_loader
        cached_gaussians = self._precompute_teacher_gaussians()
        wrapped_dataset = CachedGaussianDataset(source_full_dataset, cached_gaussians)

        # 划分 source train/val（训练用全部，验证用后 10%）
        from torch.utils.data import Subset
        n_total = len(wrapped_dataset)
        source_val_ratio = data_config.get('source_val_ratio', 0.1)
        n_val = max(int(n_total * source_val_ratio), 1)
        n_train = n_total  # 训练用全部数据
        val_start = n_total - n_val
        val_indices = list(range(val_start, n_total))

        self.source_loader = DataLoader(
            wrapped_dataset,
            batch_size=self.defense_batch_size,
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
        )
        self.source_val_loader = DataLoader(
            Subset(wrapped_dataset, val_indices),
            batch_size=self.defense_batch_size,
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
        )
        print(f"  ✓ Source 数据: 训练 {n_train} 样本（全部），验证 {n_val} 样本（后10%）")

        # 5. 设置优化器
        training_config = self.config['training']
        trainable_params = [p for p in self.model_mgr.model.parameters() if p.requires_grad]
        optimizer_type = training_config.get('optimizer', 'adamw')
        param_groups = self._build_optimizer_param_groups(self.model_mgr.model, training_config)
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                lr=training_config['lr'],  # default, groups may override
                weight_decay=training_config['weight_decay'],  # default, groups may override
                momentum=training_config.get('optimizer_momentum', 0.9),
            )
        else:
            self.optimizer = optim.AdamW(
                param_groups,
                lr=training_config['lr'],  # default, groups may override
                weight_decay=training_config['weight_decay'],  # default, groups may override
                betas=tuple(training_config.get('optimizer_betas', [0.9, 0.95])),
            )

        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in self.model_mgr.model.parameters())
        print(f"\n  ✓ 可训练参数: {num_trainable:,} / {num_total:,} ({num_trainable/num_total*100:.3f}%)")

        # 打印配置
        if self.use_param_noise:
            print(f"\n  参数加噪: σ={self.noise_scale_target} (warmup_steps={self.noise_warmup_steps})")
        if self.use_input_noise:
            print(f"  输入加噪: σ={self.input_noise_scale_target} (warmup_steps={self.input_noise_warmup_steps})")
        if self.use_gradient_conflict:
            print(f"  梯度冲突: weight={self.conflict_weight}, every_k={self.conflict_every_k}")
        if getattr(self, "use_feature_trap", False) and getattr(self, "feature_trap_weight", 0.0) > 0:
            ch = getattr(self, "feature_trap_channels", None)
            ch_str = "all" if not ch else str(ch)
            print(f"  Feature trap: mode={getattr(self, 'feature_trap_mode', 'spatial_var')}, "
                  f"weight={getattr(self, 'feature_trap_weight', 0.0)}, channels={ch_str}")
        if self.method == 'geotrap':
            if self.grad_clip_mode != 'norm' or self.grad_energy_cap_mult > 0:
                print(f"  Grad clip: mode={self.grad_clip_mode}, norm={self.grad_norm_clip}, "
                      f"energy_mult={self.grad_energy_cap_mult}, energy_apply_on={self.grad_energy_apply_on}")
            print(f"  Trap聚合: method={self.trap_aggregation_method}")
            if self.trap_aggregation_method in ('bottleneck', 'bottleneck_logsumexp', 'logsumexp'):
                print(f"    bottleneck_tau={self.trap_bottleneck_tau}")
            if self.trap_aggregation_method == 'pairwise_multiplicative' and len(self.trap_losses) >= 2:
                print(f"  乘法耦合: temperature={self.coupling_temperature}, use_log={self.coupling_use_log}")
            if self.use_awp:
                print(f"  AWP: rho={self.awp_rho}, weight={self.awp_weight}, every_k={self.awp_every_k}, "
                      f"param_scope={self.awp_param_scope}, exclude_bias={self.awp_exclude_bias}, "
                      f"include_frozen={getattr(self, 'awp_include_frozen', False)}, "
                      f"norm={getattr(self, 'awp_norm', 'tensor')}")
            if bool(getattr(self, 'use_head_attack_sim', False)):
                print(f"  Head-attack-sim: every_k={getattr(self, 'head_attack_sim_every_k', 1)}, "
                      f"rho={getattr(self, 'head_attack_sim_rho', None)}, "
                      f"lr={getattr(self, 'head_attack_sim_lr', None)}, "
                      f"weight={getattr(self, 'head_attack_sim_weight', 1.0)}, "
                      f"norm={getattr(self, 'head_attack_sim_norm', 'tensor')}, "
                      f"exclude_bias={getattr(self, 'head_attack_sim_exclude_bias', False)}, "
                      f"mp={getattr(self, 'head_attack_sim_mixed_precision', 'bf16')}, "
                      f"render_unlearn_w={getattr(self, 'head_attack_sim_render_unlearn_weight', 0.0)}")
            if getattr(self, "use_grad_surgery", False):
                print(f"  Grad surgery: mode={getattr(self, 'grad_surgery_mode', 'pcgrad')}, "
                      f"every_k={getattr(self, 'grad_surgery_every_k', 1)}")

        # Anti-shortcut 打印（支持 geotrap 和 naive_unlearning）
        if (self.freeze_head or self.freeze_head_bias or self.freeze_lora_targets) or (self.head_lr_mult != 1.0 or self.bias_lr_mult != 1.0):
            print(f"  Anti-shortcut: freeze_head={self.freeze_head}, freeze_head_bias={self.freeze_head_bias}, "
                  f"freeze_lora_targets={self.freeze_lora_targets}, "
                  f"head_lr_mult={self.head_lr_mult}, head_bias_lr_mult={self.head_bias_lr_mult}, bias_lr_mult={self.bias_lr_mult}")

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

    def _apply_antishortcut_freeze(self, model):
        """
        Anti-shortcut: 冻结 head(conv) 的部分/全部参数，避免 trap 主要由极少数 head 参数完成。

        说明：
        - freeze_head=True: 冻结 conv.weight + conv.bias
        - freeze_head_bias=True: 仅冻结 conv.bias
        """
        frozen = 0
        if self.freeze_head:
            for name, param in model.named_parameters():
                if name.startswith('conv.'):
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen += 1
                        print(f"  ✓ Anti-shortcut 冻结: {name}")
        elif self.freeze_head_bias:
            for name, param in model.named_parameters():
                if name == 'conv.bias' and param.requires_grad:
                    param.requires_grad = False
                    frozen += 1
                    print(f"  ✓ Anti-shortcut 冻结: {name}")

        if frozen == 0:
            print("  ⚠ Anti-shortcut: 未冻结任何参数（可能 head 已被冻结或名称不匹配）")

    def _apply_lora_target_freeze(self, model):
        """
        Anti-shortcut: 冻结 LoRA 可触及的层（qkv/proj），迫使 trap 写入上游 conv 层。

        说明：
        - 当 freeze_lora_targets=True 时，冻结所有名字包含 lora_target_modules 的参数
        - 默认 lora_target_modules=['qkv', 'proj']，对应 attention 的输入/输出投影
        - 这样 LoRA 攻击者无法直接修改被防御的层，需要通过上游 conv 层绕过
        """
        frozen = 0
        if self.freeze_lora_targets:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 检查参数名是否包含任何 LoRA 目标模块
                    # 匹配模式：attn.qkv.weight, attn.proj.bias 等
                    for target_module in self.lora_target_modules:
                        if f'.{target_module}.' in name or name.endswith(f'.{target_module}'):
                            param.requires_grad = False
                            frozen += 1
                            print(f"  ✓ 冻结 LoRA 目标层: {name}")
                            break

        if self.freeze_lora_targets and frozen == 0:
            print(f"  ⚠ Anti-shortcut (LoRA targets): 未冻结任何参数（可能名称不匹配）")
            print(f"  目标模块: {self.lora_target_modules}")
        elif frozen > 0:
            print(f"  ✓ 共冻结 {frozen} 个 LoRA 目标层参数")

    def _build_optimizer_param_groups(self, model, training_config: Dict[str, Any]):
        """
        构建带 LR multiplier 的 optimizer param groups（避免 head/bias 走捷径）。

        不改变默认行为：当所有 multiplier=1 且不冻结时，相当于单一 param list。
        """
        base_lr = float(training_config['lr'])
        base_wd = float(training_config.get('weight_decay', 0.0))

        head_params = []
        head_bias_params = []
        bias_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('conv.'):
                if name.endswith('.bias'):
                    head_bias_params.append(param)
                else:
                    head_params.append(param)
            elif name.endswith('.bias'):
                bias_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append({'params': other_params, 'lr': base_lr, 'weight_decay': base_wd})
        if bias_params:
            groups.append({'params': bias_params, 'lr': base_lr * self.bias_lr_mult, 'weight_decay': base_wd})
        if head_params:
            groups.append({'params': head_params, 'lr': base_lr * self.head_lr_mult, 'weight_decay': base_wd})
        if head_bias_params:
            groups.append({'params': head_bias_params, 'lr': base_lr * self.head_bias_lr_mult, 'weight_decay': base_wd})

        if not groups:
            raise ValueError("没有任何可训练参数（可能所有层都被冻结）。请检查 defense.target_layers / antishortcut 配置。")
        return groups

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
            try:
                cached_gaussians = torch.load(cache_path, map_location='cpu', weights_only=True)
            except Exception as e:
                print(f"  ⚠ 缓存文件读取失败，将重新计算: {cache_path}")
                print(f"    原因: {type(e).__name__}: {e}")
                try:
                    cache_path.unlink()
                    print("    已删除损坏缓存文件")
                except Exception:
                    pass
                cached_gaussians = None
            dataset_len = len(self.source_loader.dataset)
            if cached_gaussians is not None and len(cached_gaussians) == dataset_len:
                print(f"  ✓ 缓存命中: {len(cached_gaussians)} 个样本")
                return cached_gaussians
            if cached_gaussians is not None:
                print(f"  ⚠ 缓存样本数不匹配 ({len(cached_gaussians)} vs {dataset_len})，重新计算")

        # 缓存未命中，临时加载教师模型执行推理
        print("  缓存未命中，加载教师模型进行推理...")
        from models import ModelManager
        teacher_mgr = ModelManager(self.config)
        teacher_mgr.setup(device=self.device)
        teacher_model = teacher_mgr.model
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        dataset = self.source_loader.dataset
        temp_loader = DataLoader(
            dataset,
            batch_size=self.defense_batch_size,
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
        tmp_path = cache_path.parent / f"{cache_path.name}.{os.getpid()}.tmp"
        try:
            torch.save(cached_gaussians, tmp_path)
            tmp_path.replace(cache_path)
            print(f"  ✓ 预计算完成: {len(cached_gaussians)} 个样本，已缓存到 {cache_path}")
        except Exception as e:
            print(f"  ⚠ 预计算完成但写入缓存失败，将继续使用内存结果（下次可能需要重算）")
            print(f"    cache_path: {cache_path}")
            print(f"    原因: {type(e).__name__}: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
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

        # Teacher-gaussian cache can be multi-GB. Avoid duplicating it across experiments by
        # defaulting to a shared cache directory under project output/workspace/cache.
        #
        # You can override this via:
        #   misc:
        #     teacher_cache_dir: /path/to/cache
        misc_cfg = self.config.get('misc', {}) or {}
        override_dir = misc_cfg.get('teacher_cache_dir') or misc_cfg.get('cache_dir')
        if override_dir:
            cache_dir = Path(str(override_dir))
        else:
            project_root = Path(__file__).resolve().parent.parent
            cache_dir = project_root / 'output' / 'workspace' / 'cache'
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
        abs_diff = diff.abs()

        # Optional: per-attribute weighting (leverages the physical meaning of Gaussian vectors).
        # Useful when the trap focuses on one attribute (e.g., opacity collapse) and we want to
        # preserve source rendering quality by enforcing stronger distillation on that channel.
        w_cfg = (self.defense_config.get('distill_channel_weights') or
                 self.defense_config.get('distill_attribute_weights') or {})
        weights = None
        if isinstance(w_cfg, dict) and w_cfg:
            # group-wise weights (default 1.0)
            w_pos = float(w_cfg.get('position', 1.0))
            w_opacity = float(w_cfg.get('opacity', 1.0))
            w_scale = float(w_cfg.get('scale', 1.0))
            w_rot = float(w_cfg.get('rotation', 1.0))
            w_color = float(w_cfg.get('color', 1.0))

            w = torch.ones((14,), device=abs_diff.device, dtype=abs_diff.dtype)
            w[0:3] *= w_pos
            w[3:4] *= w_opacity
            w[4:7] *= w_scale
            w[7:11] *= w_rot
            w[11:14] *= w_color
            weights = w.view(1, 1, 14)
        elif isinstance(w_cfg, (list, tuple)) and len(w_cfg) == 14:
            w = torch.tensor([float(x) for x in w_cfg], device=abs_diff.device, dtype=abs_diff.dtype)
            weights = w.view(1, 1, 14)

        if weights is None:
            return torch.mean(abs_diff ** order)

        # Weight the per-dimension contribution.
        return torch.mean((abs_diff ** order) * weights)

    def compute_trap_loss(self, gaussians, model, input_images=None):
        """
        计算陷阱损失（制造几何陷阱）

        2+ 个 trap 自动使用两两乘法耦合 ∑_{i<j} -((1-L_i)(1-L_j)-1)，单个 trap 直接返回。
        两两耦合相比全局 ∏(1-L_i) 的优势：梯度分布更均匀，每个 trap 被独立推深。

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
        static_loss_tensors = {}
        for name, trap_loss_fn in self.trap_losses.items():
            loss = trap_loss_fn(gaussians)
            # 仅做 NaN/Inf 清理，不做 clamp 限制，让 trap 自由增长
            # 通过梯度裁剪（gradient_clip）控制训练稳定性
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            loss_dict[name] = loss.item()

            weight = float(self.trap_weights.get(name, 1.0)) if hasattr(self, "trap_weights") else 1.0
            static_loss_tensors[name] = loss * weight

        # 2. 静态 trap 聚合
        loss_list = list(static_loss_tensors.values())
        if len(loss_list) == 0:
            static_combined = torch.zeros(1, device=gaussians.device, dtype=gaussians.dtype)
        elif len(loss_list) == 1:
            static_combined = loss_list[0]
        else:
            method = getattr(self, "trap_aggregation_method", "mean")
            if method == 'sum':
                static_combined = torch.stack(loss_list).sum()
            elif method == 'mean':
                static_combined = torch.stack(loss_list).mean()
            elif method in ('bottleneck', 'bottleneck_logsumexp', 'logsumexp'):
                tau = float(getattr(self, "trap_bottleneck_tau", 0.25))
                tau = max(tau, 1e-6)
                stacked = torch.stack(loss_list)
                # tau * logsumexp(L/tau) 近似 max(L)，最小化它会持续推深”最弱 trap”
                static_combined = tau * torch.logsumexp(stacked / tau, dim=0)
            elif method == 'max':
                static_combined = torch.stack(loss_list).max()
            else:
                # 旧默认：两两乘法耦合（pairwise multiplicative coupling）
                pairwise_sum = torch.zeros(1, device=gaussians.device, dtype=gaussians.dtype)
                count = 0

                # 可选：对loss先做log变换（将(-∞,0)映射到更小范围）
                if self.coupling_use_log:
                    # log(1 + |L|) 变换，保持负号
                    transformed_losses = []
                    for loss in loss_list:
                        sign = torch.sign(loss)
                        transformed = sign * torch.log(1.0 + torch.abs(loss))
                        transformed_losses.append(transformed)
                    loss_list = transformed_losses

                # 温度缩放 + 乘法耦合
                for i in range(len(loss_list)):
                    for j in range(i + 1, len(loss_list)):
                        # 温度缩放：L/T
                        li_scaled = loss_list[i] / self.coupling_temperature
                        lj_scaled = loss_list[j] / self.coupling_temperature
                        # 乘法耦合
                        pairwise_sum = pairwise_sum - ((1.0 - li_scaled) * (1.0 - lj_scaled) - 1.0)
                        count += 1

                static_combined = pairwise_sum / max(count, 1)

        loss_dict['static_combined'] = static_combined.item()
        total_loss += static_combined

        # 3. 动态敏感度损失
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

    def _compute_gradient_conflict(self, gaussians, model):
        """
        计算特征空间梯度冲突正则

        在 model.conv（U-Net 输出 → Gaussian 参数的 1×1 卷积）的输入特征上，
        计算每对 trap loss 的梯度 cosine similarity，最小化之（推向 -1）。

        原理：不同 trap 作用于 Gaussian 的不同切片（pos/opacity/scale/...），
        输出空间梯度天然正交，无法产生冲突。但 conv 输入是 U-Net 的共享特征瓶颈，
        在此处制造冲突意味着 U-Net 无法同时满足多个 trap，攻击者微调时也面临同样困境。

        相比全参数空间梯度冲突（需要 N 次完整 backward），特征空间冲突只需从
        gaussians 反传到 conv 输入（一层 1×1 卷积），计算量极小。

        Args:
            gaussians: Gaussian 参数 [B, N, 14]，必须保留计算图
            model: LGM 模型（需要 model.conv 和 model._conflict_features）

        Returns:
            conflict_loss: 标量，mean cosine similarity（越负越好）
            conflict_info: dict，各对 trap 的 cos_sim 值
        """
        if len(self.trap_losses) < 2:
            return torch.tensor(0.0, device=gaussians.device), {}

        # 获取 hook 捕获的 conv 输入特征
        features = model._conflict_features
        if features is None:
            return torch.tensor(0.0, device=gaussians.device), {}

        # 计算每个 trap 对 features 的梯度
        trap_grads = {}
        for name, trap_fn in self.trap_losses.items():
            loss = trap_fn(gaussians)
            # 仅做 NaN/Inf 清理，不做 clamp 限制
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            grad = torch.autograd.grad(
                loss, features,
                create_graph=True, retain_graph=True,
            )[0]  # [B*4, 14, h, w]
            trap_grads[name] = grad.reshape(-1).float()  # 展平 + 转 f32 保证精度

        # 两两 cosine similarity，最小化（推向 -1）
        names = list(trap_grads.keys())
        conflict_loss = torch.zeros(1, device=gaussians.device, dtype=torch.float32)
        count = 0
        conflict_info = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cos = torch.nn.functional.cosine_similarity(
                    trap_grads[names[i]].unsqueeze(0),
                    trap_grads[names[j]].unsqueeze(0),
                )
                conflict_loss = conflict_loss + cos
                conflict_info[f'cos_{names[i]}_vs_{names[j]}'] = cos.item()
                count += 1

        conflict_loss = conflict_loss / count
        conflict_info['conflict_mean_cos'] = conflict_loss.item()

        return conflict_loss.squeeze(), conflict_info

    def _compute_feature_trap_loss(self, model):
        """
        Feature-space trap loss (first-order, render-free).

        Operates on the conv input features (UNet output) captured by the forward hook:
          model._conflict_features: [B*V, C=14, H, W]

        Intuition:
          - The Gaussian head (conv) is a 1×1 conv; it cannot create spatial structure if
            UNet outputs are spatially collapsed.
          - Collapsing spatial variability for target inputs reduces the effectiveness of
            head-only recovery fine-tuning (a common quick-attack shortcut).

        Returns:
            (loss_tensor, info_dict)
        """
        features = getattr(model, "_conflict_features", None)
        if features is None:
            loss = torch.tensor(0.0, device=self.device)
            return loss, {"feature_trap_missing": 1.0}

        f = features.float()

        ch = getattr(self, "feature_trap_channels", None)
        if ch:
            try:
                f = f[:, ch]
            except Exception:
                # Best-effort only; fall back to all channels.
                pass

        mode_raw = str(getattr(self, "feature_trap_mode", "spatial_var")).lower().strip()
        modes = [m.strip() for m in mode_raw.split("+") if m.strip()]
        if not modes:
            loss = torch.tensor(0.0, device=self.device)
            return loss, {"feature_trap_missing_mode": 1.0}

        eps = 1e-8
        total_loss = None
        info = {}

        for mode in modes:
            if mode in ("spatial_var", "var", "variance"):
                var_hw = f.var(dim=(-2, -1), unbiased=False)
                loss = var_hw.mean()
                info["feature_spatial_var"] = float(loss.detach().item())

            elif mode in ("spatial_var_rel", "spatial_var_norm", "spatial_var_normalized", "spatial_var_normalize"):
                # Scale-invariant spatial collapse:
                #   E[ Var_hw(f) / (E_hw|f|^2 + eps) ]
                # Avoids the trivial "just shrink amplitudes" workaround.
                var_hw = f.var(dim=(-2, -1), unbiased=False)
                mean_abs = f.abs().mean(dim=(-2, -1)).clamp_min(eps)
                loss = (var_hw / (mean_abs * mean_abs + eps)).mean()
                info["feature_spatial_var_rel"] = float(loss.detach().item())

            elif mode in ("channel_var", "channel_variance"):
                var_c = f.var(dim=1, unbiased=False)  # [B*V, H, W]
                loss = var_c.mean()
                info["feature_channel_var"] = float(loss.detach().item())

            elif mode in ("channel_var_rel", "channel_var_norm", "channel_var_normalized", "channel_var_normalize"):
                # Scale-invariant channel collapse:
                #   E[ Var_c(f) / (E_c|f|^2 + eps) ]
                var_c = f.var(dim=1, unbiased=False)
                mean_abs_c = f.abs().mean(dim=1).clamp_min(eps)
                loss = (var_c / (mean_abs_c * mean_abs_c + eps)).mean()
                info["feature_channel_var_rel"] = float(loss.detach().item())

            elif mode in ("spatial_mse", "demeaned_l2", "centered_l2", "spatial_l2"):
                mean_hw = f.mean(dim=(-2, -1), keepdim=True)
                loss = ((f - mean_hw) ** 2).mean()
                info["feature_spatial_mse"] = float(loss.detach().item())

            elif mode in ("l2", "energy", "mse"):
                loss = (f ** 2).mean()
                info["feature_l2"] = float(loss.detach().item())

            else:
                raise ValueError(
                    f"feature_trap.mode 不支持: {mode}（支持 spatial_var / spatial_var_rel / "
                    f"channel_var / channel_var_rel / spatial_mse / l2；也支持用 '+' 组合）"
                )

            total_loss = loss if total_loss is None else (total_loss + loss)

        if total_loss is None:
            total_loss = torch.tensor(0.0, device=self.device)
        return total_loss, info

    def _select_awp_params(self, model, include_frozen: bool = False) -> List[tuple[str, torch.nn.Parameter]]:
        """
        选择 AWP 要扰动的参数集合（默认 head: conv.*）。
        """
        scope = getattr(self, "awp_param_scope", "head")
        exclude_bias = bool(getattr(self, "awp_exclude_bias", True))

        selected: List[tuple[str, torch.nn.Parameter]] = []
        target_layers = getattr(self, "target_layers", None) or []
        for name, param in model.named_parameters():
            if (not include_frozen) and (not param.requires_grad):
                continue
            if exclude_bias and name.endswith(".bias"):
                continue

            ok = False
            if scope == "trainable":
                ok = True
            elif scope in ("target_layers", "target_layer", "selected_layers", "selected"):
                # Match the same substring semantics as selective fine-tuning.
                # This keeps AWP perturbation aligned with what we actually update.
                for layer_key in target_layers:
                    if layer_key and (layer_key in name):
                        ok = True
                        break
            elif scope == "unet":
                ok = name.startswith("unet.")
            elif scope in ("head", "conv", "gaussian_head"):
                ok = name.startswith("conv.")
            else:
                # default: head (backward compatible)
                ok = name.startswith("conv.")

            if ok:
                selected.append((name, param))
        return selected

    def _compute_awp_trap_loss(self, model, input_images):
        """
        一阶 AWP：在参数空间做一次“最大化 trap loss”的扰动，再最小化扰动点的 trap loss。

        目标近似：
            min_θ  [ L_trap(θ) + weight * L_trap(θ + ε*(θ)) ]
        其中 ε 沿着 +∇_θ L_trap 的方向（使 trap 变浅），从而增强对修复微调的鲁棒性。
        """
        include_frozen = bool(getattr(self, "awp_include_frozen", False))
        awp_params = self._select_awp_params(model, include_frozen=include_frozen)
        if not awp_params:
            return {}, torch.tensor(0.0, device=input_images.device)

        # Temporarily enable gradients for selected params (e.g., frozen head) to obtain
        # the perturbation direction. We do a fresh forward to guarantee the graph includes them.
        req_backup = {name: p.requires_grad for name, p in awp_params}
        try:
            for _, p in awp_params:
                if not p.requires_grad:
                    p.requires_grad_(True)

            with torch.set_grad_enabled(True):
                with self._autocast_defense():
                    gaussians_base = model.forward_gaussians(input_images)
                gaussians_base = gaussians_base.float()
            _, base_trap_loss = self.compute_trap_loss(gaussians_base, model)

            params_only = [p for _, p in awp_params]
            grads = torch.autograd.grad(
                outputs=base_trap_loss,
                inputs=params_only,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        finally:
            for name, p in awp_params:
                p.requires_grad_(req_backup.get(name, p.requires_grad))

        # Apply perturbation (first-order):
        #   theta <- theta + rho * g / ||g||   (norm can be per-tensor or global)
        backup = {}
        rho = float(getattr(self, "awp_rho", 0.0))
        eps = 1e-12
        norm_mode = str(getattr(self, "awp_norm", "tensor")).lower()
        if norm_mode in ("global", "all", "total"):
            total = None
            for grad in grads:
                if grad is None:
                    continue
                g2 = grad.detach().float().pow(2).sum()
                total = g2 if total is None else (total + g2)
            if total is None:
                total = torch.tensor(0.0, device=input_images.device)
            global_norm = torch.sqrt(total + eps).clamp_min(eps)

            for (name, param), grad in zip(awp_params, grads):
                if grad is None:
                    continue
                g = grad.detach()
                delta = (rho * g / global_norm).to(dtype=param.data.dtype)
                backup[name] = param.data.clone()
                param.data.add_(delta)
        else:
            # Legacy: per-tensor normalize (safe only when selecting few tensors)
            for (name, param), grad in zip(awp_params, grads):
                if grad is None:
                    continue
                g = grad.detach()
                norm = g.float().norm().clamp_min(eps)
                delta = (rho * g / norm).to(dtype=param.data.dtype)
                backup[name] = param.data.clone()
                param.data.add_(delta)

        # Forward at perturbed weights
        with torch.set_grad_enabled(True):
            with self._autocast_defense():
                gaussians_awp = model.forward_gaussians(input_images)
            gaussians_awp = gaussians_awp.float()

        awp_dict, awp_loss = self.compute_trap_loss(gaussians_awp, model)

        # Restore weights
        for name, param in awp_params:
            if name in backup:
                param.data.copy_(backup[name])

        return awp_dict, awp_loss

    def _update_noise_scale(self, global_step: int):
        """
        根据当前训练步数更新噪声scale（线性warmup）

        Args:
            global_step: 当前全局优化器步数
        """
        # 更新参数噪声scale
        if self.noise_warmup_steps > 0:
            # 线性warmup: 从0增长到target值
            warmup_progress = min(1.0, global_step / self.noise_warmup_steps)
            self.current_noise_scale = self.noise_scale_target * warmup_progress
        else:
            # 无warmup，直接使用目标值
            self.current_noise_scale = self.noise_scale_target

        # 更新输入噪声scale
        if self.input_noise_warmup_steps > 0:
            warmup_progress = min(1.0, global_step / self.input_noise_warmup_steps)
            self.current_input_noise_scale = self.input_noise_scale_target * warmup_progress
        else:
            self.current_input_noise_scale = self.input_noise_scale_target

    def _add_param_noise(self, model):
        """
        对模型参数添加高斯噪声，返回原始权重备份

        Args:
            model: 模型

        Returns:
            original_state: {name: tensor} 原始权重备份
        """
        original_state = {}
        for name, param in model.named_parameters():
            original_state[name] = param.data.clone()
            noise = torch.randn_like(param) * self.current_noise_scale
            param.data.add_(noise)
        return original_state

    def _restore_params(self, model, original_state):
        """恢复模型参数到原始权重"""
        for name, param in model.named_parameters():
            param.data.copy_(original_state[name])

    def _add_input_noise(self, input_images):
        """
        对输入图像添加高斯噪声

        输入组成 [B, 4, 9, H, W]:
        - 前3通道: ImageNet归一化的RGB图像，范围约[-2.1, 2.6]
        - 后6通道: Rays Plucker坐标，范围约[-1, 1]

        噪声会同时作用于RGB和rays，模拟输入扰动。

        Args:
            input_images: 输入图像 [B, 4, 9, H, W]

        Returns:
            noisy_images: 加噪后的图像
        """
        noise = torch.randn_like(input_images) * self.current_input_noise_scale
        noisy_images = input_images + noise
        return noisy_images

    def train_step(self, batch, is_target_data=True):
        """
        训练一个 step（只计算损失和反向传播，不更新参数）

        Target 数据：GeoTrap 默认对目标输入做一次 trap 前向；
                   若开启 input_noise，则主 trap 通路直接使用加噪输入。
        Source 数据：干净权重上前向 → 蒸馏 loss → backward

        Args:
            batch: 数据批次
            is_target_data: 是否为 target 数据（True=陷阱/遗忘损失，False=蒸馏损失）

        Returns:
            loss_dict: 损失字典
        """
        self.model_mgr.model.train()
        model = self.model_mgr.model

        # 移动数据到设备
        input_images = batch['input_images'].to(self.device)  # [B, 4, 9, H, W]

        loss_dict = {}

        if is_target_data:
            if self.method == 'naive_unlearning':
                # Naive Unlearning: 对 target 数据的渲染 loss 做梯度上升（干净权重）
                with self._autocast_defense():
                    student_gaussians = model.forward_gaussians(input_images)
                student_gaussians = student_gaussians.float()
                total_loss = self._compute_naive_unlearning_loss(batch, loss_dict)
            else:
                # GeoTrap:
                # - 默认: 干净输入上的 trap loss
                # - 若开启 input_noise: 主 trap 通路直接改为加噪输入，不再额外叠加 clean/noisy 两套输入图
                # - 若开启 param_noise: 仍会额外做一条参数加噪通路
                lambda_trap = self.defense_config.get('lambda_trap', 1.0)
                trap_input_images = input_images
                if self.use_input_noise:
                    trap_input_images = self._add_input_noise(input_images)

                # (a) 主 trap 前向：默认用干净输入；开启 input_noise 后改为加噪输入
                with torch.set_grad_enabled(True):
                    with self._autocast_defense():
                        gaussians_clean = model.forward_gaussians(trap_input_images)
                    gaussians_clean = gaussians_clean.float()

                trap_dict_clean, trap_loss_clean = self.compute_trap_loss(
                    gaussians_clean, model)
                loss_dict.update(trap_dict_clean)

                total_loss = lambda_trap * trap_loss_clean

                # (a.1) Feature-space trap（可选）：在 UNet→head 的共享瓶颈特征上制造空间塌缩
                # 目标：让 head-only 的恢复微调不再能快速修复结构（conv 是 1×1）。
                if getattr(self, "use_feature_trap", False) and getattr(self, "feature_trap_weight", 0.0) > 0:
                    ft_loss, ft_info = self._compute_feature_trap_loss(model)
                    loss_dict.update(ft_info)
                    total_loss = total_loss + self.feature_trap_weight * ft_loss

                # (a.2) 特征空间梯度冲突（在干净前向的计算图上）
                self._train_step_counter += 1
                if (self.use_gradient_conflict
                        and len(self.trap_losses) >= 2
                        and self._train_step_counter % self.conflict_every_k == 0):
                    conflict_loss, conflict_info = self._compute_gradient_conflict(
                        gaussians_clean, model)
                    loss_dict.update(conflict_info)
                    total_loss = total_loss + self.conflict_weight * conflict_loss

                # (a.3) AWP：对“可修复方向”的一阶鲁棒性（可选）
                if (self.use_awp
                        and self.awp_every_k > 0
                        and self._train_step_counter % self.awp_every_k == 0):
                    awp_dict, awp_loss = self._compute_awp_trap_loss(
                        model, input_images
                    )
                    for k, v in awp_dict.items():
                        loss_dict[f'awp_{k}'] = v
                    total_loss = total_loss + lambda_trap * self.awp_weight * awp_loss

                # (a.4) Head-attack simulation（可选）：
                # 通过 render_loss 的梯度方向模拟攻击者对 head(conv) 的一步更新，
                # 然后要求在该 perturbed head 上 trap 仍然成立，压制 "head-only recovery" shortcut。
                if bool(getattr(self, 'use_head_attack_sim', False)):
                    hs_dict, hs_loss = self._compute_head_attack_sim_trap_loss(
                        model, batch, input_images
                    )
                    if hs_dict:
                        loss_dict.update(hs_dict)
                    if isinstance(hs_loss, torch.Tensor):
                        total_loss = total_loss + lambda_trap * float(getattr(self, 'head_attack_sim_weight', 1.0)) * hs_loss

                # (b) 加噪权重前向 → trap loss（参数鲁棒性）
                if self.use_param_noise:
                    original_state = self._add_param_noise(model)

                    with torch.set_grad_enabled(True):
                        with self._autocast_defense():
                            gaussians_noisy = model.forward_gaussians(trap_input_images)
                        gaussians_noisy = gaussians_noisy.float()

                    trap_dict_noisy, trap_loss_noisy = self.compute_trap_loss(
                        gaussians_noisy, model)
                    # 记录加噪版本的指标（带 param_noisy_ 前缀）
                    for k, v in trap_dict_noisy.items():
                        loss_dict[f'param_noisy_{k}'] = v

                    total_loss = total_loss + lambda_trap * trap_loss_noisy

                    # 恢复干净权重（backward 仍基于两份计算图）
                    self._restore_params(model, original_state)

        else:
            # Source Data: 干净权重上前向 → 蒸馏损失
            with torch.set_grad_enabled(True):
                with self._autocast_defense():
                    student_gaussians = model.forward_gaussians(input_images)
                student_gaussians = student_gaussians.float()

            teacher_gaussians = batch['teacher_gaussians'].to(self.device)
            distill_loss = self.compute_distillation_loss(student_gaussians, teacher_gaussians)
            loss_dict['distillation'] = distill_loss.item()

            lambda_distill = self.defense_config.get('lambda_distill', 1.0)
            total_loss = lambda_distill * distill_loss

        # Optional: anchoring regularizer (stage-wise)
        anchor_loss = self._compute_anchor_loss(model, is_target_data=is_target_data)
        if anchor_loss is not None:
            loss_dict['anchor'] = float(anchor_loss.detach().float().item())
            total_loss = total_loss + float(self.anchor_weight) * anchor_loss

        loss_dict['loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        # 反向传播（梯度累积，不立即更新参数）
        if isinstance(total_loss, torch.Tensor):
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

        with self._autocast_defense():
            results = self.model_mgr.model.forward(data, step_ratio=1.0)

        render_loss = results['loss']
        loss_dict['render_loss'] = render_loss.item()

        # 取负：最小化 -render_loss = 最大化 render_loss（梯度上升）
        lambda_trap = self.defense_config.get('lambda_trap', 1.0)
        total_loss = -lambda_trap * render_loss

        loss_dict['unlearning_loss'] = total_loss.item()
        return total_loss

    def train_epoch(self, epoch: int, global_step: int = 0, step_callback=None, max_steps: int | None = None):
        """
        训练一个 epoch（双数据加载器模式 + 梯度累积）

        Args:
            epoch: 当前 epoch 编号
            global_step: 全局优化器步数起始值（跨 epoch 累计，每次 optimizer.step() 时 +1）
            step_callback: 每个优化器步后的回调 callable(global_step, loss_dict) -> None
                           可用于周期性评估等外部逻辑
            max_steps: 若不为 None，则本 epoch 最多训练到优化器步数 < max_steps。
                       用于 step-based 训练避免最后一个 epoch 超出目标步数。

        Returns:
            avg_metrics: 平均损失字典
            global_step: 更新后的全局优化器步数
        """
        import time
        self.model_mgr.model.train()

        total_losses = {}
        num_batches = 0
        accumulation_counter = 0  # 梯度累积计数器
        accum_has_target = False
        accum_has_source = False

        # 时间跟踪
        epoch_start_time = time.time()
        step_times = []  # 记录最近的步时间用于平滑估计

        # 双数据加载器：按比例混合source和target
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        # 计算总 batch 数（按 target 数据量定义 epoch，source 是辅助）
        # step-based 训练下，需要根据 max_steps 和梯度累积步数计算需要的 batch 数
        max_batches = len(self.target_loader)
        planned_batches = max_batches
        if max_steps is not None:
            remaining_optimizer_steps = max_steps - global_step
            # 需要的batch数 = 剩余优化器步数 × 梯度累积步数
            needed_batches = remaining_optimizer_steps * self.gradient_accumulation_steps
            planned_batches = max(0, min(max_batches, needed_batches))

        pbar = tqdm(range(planned_batches), desc=f"Epoch {epoch}")
        last_step_time = time.time()

        for batch_idx in pbar:
            # 按比例决定使用source还是target
            import random
            use_source = random.random() < self.source_ratio

            try:
                if use_source:
                    batch = next(source_iter)
                    loss_dict = self.train_step(batch, is_target_data=False)
                    accum_has_source = True
                else:
                    batch = next(target_iter)
                    loss_dict = self.train_step(batch, is_target_data=True)
                    accum_has_target = True
                    if getattr(self, "use_grad_surgery", False):
                        self._maybe_apply_grad_surgery_(source_iter)

            except StopIteration:
                if use_source:
                    source_iter = iter(self.source_loader)
                    batch = next(source_iter)
                    loss_dict = self.train_step(batch, is_target_data=False)
                    accum_has_source = True
                else:
                    target_iter = iter(self.target_loader)
                    batch = next(target_iter)
                    loss_dict = self.train_step(batch, is_target_data=True)
                    accum_has_target = True
                    if getattr(self, "use_grad_surgery", False):
                        self._maybe_apply_grad_surgery_(source_iter)

            # 累积损失
            for key, value in loss_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            num_batches += 1
            accumulation_counter += 1

            # 梯度累积：每 N 步或最后一个 batch 时更新参数
            if accumulation_counter % self.gradient_accumulation_steps == 0 or batch_idx == planned_batches - 1:
                # 梯度处理（防御阶段专用）：energy cap / norm clip
                # - energy: 限制每个参数张量的梯度能量上限，降低“少数张量主导更新”的 shortcut
                # - norm: 传统全局范数裁剪（主要用于数值稳定性）
                mode = getattr(self, 'grad_clip_mode', 'norm')
                if mode in ('energy', 'both') and getattr(self, 'grad_energy_cap_mult', 0.0) > 0:
                    apply_on = getattr(self, 'grad_energy_apply_on', 'all')
                    should_apply_energy = (
                        (apply_on == 'all')
                        or (apply_on == 'target' and accum_has_target)
                        or (apply_on == 'source' and accum_has_source)
                    )
                    if should_apply_energy:
                        from tools.utils import cap_grad_tensor_energy_
                        cap_grad_tensor_energy_(
                            self.model_mgr.model.parameters(),
                            cap_mult=self.grad_energy_cap_mult,
                            eps=1e-12,
                            return_stats=False,
                        )
                if mode in ('norm', 'both') and getattr(self, 'grad_norm_clip', 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_mgr.model.parameters(),
                        self.grad_norm_clip
                    )
                # 更新参数
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulation_counter = 0
                accum_has_target = False
                accum_has_source = False

                # 优化器步数 +1（只在实际更新参数时计数）
                global_step += 1

                # 记录效率指标
                if hasattr(self, 'config') and hasattr(self.config, '_efficiency_tracker'):
                    tracker = self.config._efficiency_tracker
                    if tracker is not None:
                        tracker.record(
                            step=global_step,
                            epoch=epoch,
                            step_time=avg_step_time if step_times else 0.0,
                            loss=loss_dict.get('loss'),
                        )

                # 更新噪声scale（warmup）
                if (self.use_param_noise and self.noise_warmup_steps > 0) or \
                   (self.use_input_noise and self.input_noise_warmup_steps > 0):
                    self._update_noise_scale(global_step)

                # 计算时间统计
                current_time = time.time()
                step_time = current_time - last_step_time
                step_times.append(step_time)
                # 保持最近20步的时间用于平滑估计
                if len(step_times) > 20:
                    step_times.pop(0)
                avg_step_time = sum(step_times) / len(step_times)
                last_step_time = current_time

                # 更新进度条（显示分项 loss + ETA）
                postfix = {'loss': f"{loss_dict['loss']:.4f}"}
                for k in ('distillation', 'static_combined'):
                    if k in loss_dict:
                        short = {'distillation': 'dist', 'static_combined': 'trap'}[k]
                        postfix[short] = f"{loss_dict[k]:.4f}"
                if 'conflict_mean_cos' in loss_dict:
                    postfix['cos'] = f"{loss_dict['conflict_mean_cos']:.3f}"
                postfix['opt_step'] = global_step  # 明确标注为优化器步数

                # 添加ETA估计
                if max_steps is not None and len(step_times) >= 3:
                    remaining_steps = max_steps - global_step
                    eta_seconds = remaining_steps * avg_step_time
                    if eta_seconds < 60:
                        postfix['ETA'] = f"{int(eta_seconds)}s"
                    elif eta_seconds < 3600:
                        postfix['ETA'] = f"{int(eta_seconds/60)}m{int(eta_seconds%60)}s"
                    else:
                        hours = int(eta_seconds / 3600)
                        minutes = int((eta_seconds % 3600) / 60)
                        postfix['ETA'] = f"{hours}h{minutes}m"

                pbar.set_postfix(postfix)

                # 步回调（只在优化器更新后调用）
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

    def _maybe_apply_grad_surgery_(self, source_iter) -> None:
        """
        Optionally apply first-order gradient surgery on current gradients (target step),
        using a fresh source batch to compute the distillation gradient direction.

        This modifies model parameter .grad in-place.
        """
        if not getattr(self, "use_grad_surgery", False):
            return
        if getattr(self, "grad_surgery_every_k", 1) <= 0:
            return

        # Reuse the existing step counter to keep schedules consistent with other components.
        step = int(getattr(self, "_train_step_counter", 0))
        if step <= 0:
            step = 1
        if step % int(getattr(self, "grad_surgery_every_k", 1)) != 0:
            return

        if str(getattr(self, "grad_surgery_mode", "pcgrad")).lower() not in ("pcgrad", "projection"):
            return

        # Sample one source batch for distillation gradient direction.
        try:
            src_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(self.source_loader)
            src_batch = next(source_iter)

        self._apply_pcgrad_against_source_(src_batch)

    def _apply_pcgrad_against_source_(self, source_batch) -> None:
        """
        PCGrad-style projection:
          If dot(g_trap, g_distill) < 0, project g_trap to be orthogonal to g_distill.

        This reduces first-order increase of the distillation loss caused by target/trap updates,
        improving retention at similar trap strength.
        """
        model = self.model_mgr.model
        device = self.device
        eps = float(getattr(self, "grad_surgery_eps", 1e-12))

        # Only consider currently trainable params (selective fine-tuning friendly).
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return

        # Compute distillation gradient direction on a single source batch (no grad accumulation here).
        input_images = source_batch.get('input_images')
        teacher_gaussians = source_batch.get('teacher_gaussians')
        if input_images is None or teacher_gaussians is None:
            return

        input_images = input_images.to(device)
        teacher_gaussians = teacher_gaussians.to(device)

        with torch.set_grad_enabled(True):
            with self._autocast_defense():
                student_gaussians = model.forward_gaussians(input_images)
            student_gaussians = student_gaussians.float()
            distill_loss = self.compute_distillation_loss(student_gaussians, teacher_gaussians)

        grads_s = torch.autograd.grad(
            outputs=distill_loss,
            inputs=params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Project current grads (trap) against distill grads.
        for p, gs in zip(params, grads_s):
            gt = p.grad
            if gt is None or gs is None:
                continue
            # Use fp32 for dot/norm computations, but apply in original dtype.
            gt_f = gt.detach().float()
            gs_f = gs.detach().float()
            dot = (gt_f * gs_f).sum()
            if not torch.isfinite(dot):
                continue
            # Conflict if gradients point in opposite directions (dot < 0).
            if dot >= 0:
                continue
            denom = gs_f.pow(2).sum().clamp_min(eps)
            scale = (dot / denom).to(dtype=gt.dtype)
            p.grad.add_((-scale) * gs.to(dtype=gt.dtype))

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
            from tools.utils import prepare_lgm_data
            with torch.no_grad():
                for batch in tqdm(self.target_val_loader, desc="Val [Target]"):
                    data = prepare_lgm_data(batch, self.model_mgr.model, self.device)
                    with self._autocast_defense():
                        results = self.model_mgr.model.forward(data, step_ratio=1.0)
                    total_losses.setdefault('render_loss', 0.0)
                    total_losses['render_loss'] += results['loss'].item()
                    num_batches += 1
        else:
            # GeoTrap: trap 效果
            with torch.no_grad():
                for batch in tqdm(self.target_val_loader, desc="Val [Target]"):
                    input_images = batch['input_images'].to(self.device)
                    with self._autocast_defense():
                        student_gaussians = self.model_mgr.model.forward_gaussians(input_images)

                    # Use the same aggregation semantics as training (static_combined/total).
                    loss_dict, _ = self.compute_trap_loss(
                        student_gaussians.float(), self.model_mgr.model
                    )

                    # Optional: feature trap stats (does not require grad; hook works in no_grad too).
                    if getattr(self, "use_feature_trap", False) and getattr(self, "feature_trap_weight", 0.0) > 0:
                        try:
                            _, ft_info = self._compute_feature_trap_loss(self.model_mgr.model)
                            loss_dict.update(ft_info)
                        except Exception:
                            pass

                    for key, value in loss_dict.items():
                        if key not in total_losses:
                            total_losses[key] = 0.0
                        total_losses[key] += value
                    num_batches += 1

        # 2. Source 验证：蒸馏质量（使用后 10% 验证集）
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

        # 3. 两两乘法耦合指标（从平均 trap loss 计算）
        trap_names = list(self.trap_losses.keys())
        if len(trap_names) >= 2:
            pairwise_sum = 0.0
            count = 0
            for i in range(len(trap_names)):
                for j in range(i + 1, len(trap_names)):
                    li = avg_metrics[trap_names[i]]
                    lj = avg_metrics[trap_names[j]]
                    pairwise_sum += -((1.0 - li) * (1.0 - lj) - 1.0)
                    count += 1
            avg_metrics['coupling_value'] = pairwise_sum / count

        # 4. 特征空间梯度冲突指标（需要梯度，取第一个 target batch 采样）
        if (self.use_gradient_conflict and len(self.trap_losses) >= 2
                and self.target_val_loader is not None):
            model = self.model_mgr.model
            model.eval()
            try:
                val_batch = next(iter(self.target_val_loader))
                input_images = val_batch['input_images'].to(self.device)
                with torch.enable_grad():
                    with self._autocast_defense():
                        gaussians = model.forward_gaussians(input_images)
                    gaussians = gaussians.float()
                    _, conflict_info = self._compute_gradient_conflict(gaussians, model)
                avg_metrics.update(conflict_info)
            except StopIteration:
                pass

        return avg_metrics

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
            # 打印分项 loss
            parts = [f"Epoch {epoch}/{num_epochs}"]
            for k in ('loss', 'distillation', 'static_combined'):
                if k in train_metrics:
                    parts.append(f"{k}={train_metrics[k]:.4f}")
            # 打印各 trap 分项
            for k, v in train_metrics.items():
                if k.endswith('_static') or k.endswith('_dynamic'):
                    parts.append(f"{k}={v:.4f}")
            print("\n  " + " | ".join(parts))

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

    def save_checkpoint(self, save_dir: str, epoch: int, is_final: bool = False):
        """
        保存检查点

        Args:
            save_dir: 保存目录
            epoch: 当前 epoch
            is_final: 是否为最终 checkpoint（用于注册）
        """
        # 为了节省磁盘空间，默认不在 save_dir 里额外保存 checkpoint / model 拷贝。
        # Pipeline/实验复用依赖的是 model_registry（tag cache），因此这里只在 final 时注册一次即可。
        if not is_final:
            return

        tag = self.defense_config.get('tag')
        if not tag:
            print(f"[DefenseTrainer] 提示: 未配置 defense.tag，跳过注册。"
                  f"可在 config 中添加 defense.tag 自动注册模型")
            return

        metadata = {
            "epoch": epoch,
            "target_layers": self.target_layers,
            "trap_losses": list(self.trap_losses.keys()),
            "defense_config": {
                k: v for k, v in self.defense_config.items()
                if k not in ('target', 'target_data')
            },
        }
        print(f"[DefenseTrainer] 注册模型到 model_registry: tag={tag}（可能需要一点时间写盘）")
        from tools.model_registry import register as registry_register
        registry_register(tag, self.model_mgr.model.state_dict(), metadata=metadata)


def load_or_train_defense(config, device='cuda', save_dir=None, cache_mode: str = "registry",
                          return_state_dict: bool = False):
    """
    一行加载或训练防御模型。

    根据配置自动计算 hash tag，如果 registry 中已存在则跳过训练，否则触发训练并注册。
    defense.method='none' 时跳过防御，返回 None。

    Args:
        config: 完整配置字典（会被修改 defense.tag）
        device: 训练设备
        save_dir: checkpoint 保存目录（默认自动生成）
        cache_mode: 防御模型缓存策略
            - "registry"（默认）: 命中则加载；未命中则训练并写入 model_registry
            - "readonly": 命中则加载；未命中则训练但不写入 model_registry（用于省磁盘）
            - "writeonly": 不读取缓存，强制重新训练并写入 model_registry（用于强制更新）
            - "none": 不读取也不写入 model_registry（每次都训练，最省磁盘但最慢）
        return_state_dict: 是否返回防御模型 state_dict（用于不落盘情况下 Phase 3 直接加载）

    Returns:
        (tag, defense_history) 或 (tag, defense_history, state_dict):
            tag — hash 字符串，用于 model_resume_override=f"tag:{tag}"；method='none' 时为 None
            defense_history — 训练历史列表，缓存命中时为 None
            state_dict — cache_mode != "registry" 或 return_state_dict=True 时可用（CPU 上的权重）
    """
    import copy
    from tools.utils import compute_defense_hash
    from tools.model_registry import REGISTRY_DIR

    method = config.get('defense', {}).get('method', 'geotrap')
    if method == 'none':
        print("[Defense] method=none，跳过防御训练")
        if return_state_dict:
            return None, None, None
        return None, None

    cache_mode = (cache_mode or "registry").lower()
    if cache_mode not in ("registry", "readonly", "none", "writeonly"):
        raise ValueError(f"cache_mode 不支持: {cache_mode}（支持 registry/readonly/none/writeonly）")

    config = copy.deepcopy(config)
    tag = compute_defense_hash(config)
    model_path = REGISTRY_DIR / tag / "model.pth"

    if cache_mode in ("registry", "readonly") and model_path.exists():
        print(f"[Defense] 缓存命中: tag={tag}")
        print(f"[Defense] 模型路径: {model_path}")
        if return_state_dict or cache_mode != "registry":
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            return tag, None, state_dict
        return tag, None

    print(f"[Defense] 缓存未命中: tag={tag}，开始训练...")
    config['defense']['tag'] = tag

    target_layers = config.get('defense', {}).get('target_layers')
    defense_epochs = config['training'].get('defense_epochs', 25)
    if defense_epochs is None:
        defense_epochs = 25
    defense_steps = config['training'].get('defense_steps')

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

    # Optional: multi-stage step schedule (curriculum).
    #
    # This is useful for the retention-vs-trap tradeoff:
    # - Stage A: plant a deep trap with target-heavy mixing / lower distill.
    # - Stage B: keep trap active but increase distill + source mixing to recover retention.
    schedule_cfg = (config.get('defense', {}).get('schedule') or
                    config.get('defense', {}).get('curriculum') or
                    config.get('defense', {}).get('stages') or {})
    if isinstance(schedule_cfg, list):
        schedule_stages = schedule_cfg
    else:
        schedule_stages = schedule_cfg.get('stages') if isinstance(schedule_cfg, dict) else None
    if not schedule_stages:
        schedule_stages = None

    # 优先使用 defense_steps（显式检查 None，避免 0 被误判为 epoch-based）
    if defense_steps is not None:
        use_steps = True
        total_steps = defense_steps
        print(f"[Defense] 训练模式: step-based, total_optimizer_steps={total_steps}")
    else:
        use_steps = False
        print(f"[Defense] 训练模式: epoch-based, defense_epochs={defense_epochs}")

    def _apply_stage_overrides(stage_cfg: dict) -> None:
        if not isinstance(stage_cfg, dict):
            return
        if 'source_ratio' in stage_cfg:
            try:
                trainer.source_ratio = float(stage_cfg['source_ratio'])
            except Exception:
                pass
        if 'lambda_trap' in stage_cfg:
            try:
                trainer.defense_config['lambda_trap'] = float(stage_cfg['lambda_trap'])
            except Exception:
                pass
        if 'lambda_distill' in stage_cfg:
            try:
                trainer.defense_config['lambda_distill'] = float(stage_cfg['lambda_distill'])
            except Exception:
                pass
        # Stage-wise anchoring controls (first-order).
        if 'anchor_enabled' in stage_cfg:
            try:
                trainer.anchor_enabled = bool(stage_cfg['anchor_enabled'])
            except Exception:
                pass
        if 'anchor_weight' in stage_cfg:
            try:
                trainer.anchor_weight = float(stage_cfg['anchor_weight'])
            except Exception:
                pass
        if 'anchor_apply_on' in stage_cfg:
            try:
                trainer.anchor_apply_on = str(stage_cfg['anchor_apply_on']).lower().strip()
            except Exception:
                pass
        if 'anchor_patterns' in stage_cfg:
            pats = stage_cfg.get('anchor_patterns')
            if isinstance(pats, str):
                pats = [p.strip() for p in pats.split(',') if p.strip()]
            if isinstance(pats, (list, tuple)):
                trainer.anchor_patterns = [str(p) for p in pats if str(p)]

    epoch = 0
    use_schedule = (schedule_stages is not None) and use_steps
    if use_schedule:
        # Validate schedule sums to total_steps (best-effort; allow mismatch but warn).
        try:
            planned = sum(int(s.get('steps', 0) or 0) for s in schedule_stages)
        except Exception:
            planned = None
        if planned is not None and planned != int(total_steps):
            # Prefer the explicit `training.defense_steps` (often overridden via CLI) and
            # scale the stage steps proportionally, to keep the schedule usable at
            # different total budgets.
            print(f"[Defense][Schedule] 警告: stages 总步数={planned} 与 training.defense_steps={total_steps} 不一致。"
                  f"将按比例缩放 stages 以匹配 defense_steps。")
            if planned <= 0:
                print("[Defense][Schedule] stages 总步数无效（<=0），已忽略 schedule。")
                schedule_stages = None
            else:
                # Collect non-zero stages to keep the schedule shape.
                nonzero = []
                for idx, stage in enumerate(schedule_stages):
                    try:
                        s = int(stage.get('steps', 0) or 0)
                    except Exception:
                        s = 0
                    if s > 0:
                        nonzero.append((idx, s))

                if not nonzero:
                    print("[Defense][Schedule] stages 中无有效 steps（全部<=0），已忽略 schedule。")
                    schedule_stages = None
                else:
                    tgt = int(total_steps)
                    if tgt <= 0:
                        print("[Defense][Schedule] defense_steps<=0，已忽略 schedule。")
                        schedule_stages = None
                    elif tgt < len(nonzero):
                        print(f"[Defense][Schedule] 警告: defense_steps={tgt} 小于非零 stage 数={len(nonzero)}，"
                              f"将仅保留前 {tgt} 个 stage，每个 1 step。")
                        for j, (idx, _s) in enumerate(nonzero):
                            schedule_stages[idx]['steps'] = 1 if j < tgt else 0
                    else:
                        scale = float(tgt) / float(planned)
                        new_steps = []
                        for idx, s in nonzero:
                            scaled = int(round(float(s) * scale))
                            scaled = max(1, scaled)
                            new_steps.append([idx, scaled])

                        # Fix rounding drift to make sum exactly `tgt`.
                        current = sum(s for _idx, s in new_steps)
                        diff = tgt - current
                        if diff != 0:
                            # First, add all positive diff to the last stage.
                            if diff > 0:
                                new_steps[-1][1] += diff
                            else:
                                # Subtract from stages with steps>1, from the end.
                                need = -diff
                                for k in range(len(new_steps) - 1, -1, -1):
                                    if need <= 0:
                                        break
                                    can = max(0, new_steps[k][1] - 1)
                                    take = min(can, need)
                                    new_steps[k][1] -= take
                                    need -= take
                                if need > 0:
                                    # As a last resort, force the last stage to absorb the remainder.
                                    new_steps[-1][1] = max(1, new_steps[-1][1] - need)

                        # Apply scaled steps back.
                        for idx, scaled in new_steps:
                            schedule_stages[idx]['steps'] = int(scaled)

        use_schedule = schedule_stages is not None

    if use_schedule:
        stage_idx = 0
        for stage in schedule_stages:
            stage_idx += 1
            stage_steps = int(stage.get('steps', 0) or 0)
            if stage_steps <= 0:
                continue
            stage_name = str(stage.get('name', f'stage{stage_idx}'))
            # Optional: snapshot anchor state at the *start* of this stage (typically stage B).
            if bool(stage.get('anchor_snapshot', False)):
                pats = stage.get('anchor_patterns', None)
                if isinstance(pats, str):
                    pats = [p.strip() for p in pats.split(',') if p.strip()]
                if pats is None:
                    pats = getattr(trainer, 'anchor_patterns', None)
                try:
                    n = trainer.snapshot_anchor_state(patterns=pats)
                    print(f"[Defense][Schedule] Anchor snapshot taken: tensors={n}")
                except Exception as e:
                    print(f"[Defense][Schedule] Anchor snapshot failed: {type(e).__name__}: {e}")
            _apply_stage_overrides(stage)
            stage_end = global_step + stage_steps
            print(f"[Defense][Schedule] Stage {stage_idx}/{len(schedule_stages)} '{stage_name}': "
                  f"steps={stage_steps}, source_ratio={getattr(trainer, 'source_ratio', None)}, "
                  f"lambda_trap={trainer.defense_config.get('lambda_trap')}, "
                  f"lambda_distill={trainer.defense_config.get('lambda_distill')}, "
                  f"anchor_w={getattr(trainer, 'anchor_weight', 0.0)}")

            while global_step < stage_end:
                epoch += 1
                train_metrics, global_step = trainer.train_epoch(
                    epoch,
                    global_step,
                    max_steps=stage_end,
                )
                combined = {f"train_{k}": v for k, v in train_metrics.items()}
                combined['epoch'] = epoch
                combined['schedule_stage'] = stage_name
                combined['schedule_stage_idx'] = stage_idx
                combined['schedule_stage_end'] = stage_end

                do_val = (epoch % validate_every == 0) or (global_step >= stage_end)
                if do_val:
                    val_metrics = trainer.validate()
                    combined.update({f"val_{k}": v for k, v in val_metrics.items()})
                    print(f"  [Defense] Epoch {epoch} (Stage {stage_idx}, Optimizer Step {global_step}/{stage_end}) - "
                          f"Loss: {train_metrics['loss']:.4f}, "
                          f"DistillMSE: {val_metrics.get('source_distill_mse', 0):.6f}")
                    for k in ('position_static', 'scale_static', 'opacity_static',
                              'rotation_static', 'color_static', 'coupling_value',
                              'conflict_mean_cos'):
                        if k in val_metrics:
                            print(f"    {k}: {val_metrics[k]:.4f}")
                else:
                    print(f"  [Defense] Epoch {epoch} (Stage {stage_idx}, Optimizer Step {global_step}/{stage_end}) - "
                          f"Loss: {train_metrics['loss']:.4f}")

                epoch_history.append(combined)

        # Final checkpoint after all stages
        if cache_mode in ("registry", "writeonly"):
            trainer.save_checkpoint(save_dir, epoch, is_final=True)

    else:
        # Legacy single-stage training loop
        while True:
            if use_steps and global_step >= total_steps:
                # 防止在恢复/异常情况下重复多跑一步
                break
            epoch += 1
            train_metrics, global_step = trainer.train_epoch(
                epoch,
                global_step,
                max_steps=total_steps if use_steps else None,
            )
            combined = {f"train_{k}": v for k, v in train_metrics.items()}
            combined['epoch'] = epoch

            do_val = (epoch % validate_every == 0) or (use_steps and global_step >= total_steps) or (not use_steps and epoch == defense_epochs)
            if do_val:
                val_metrics = trainer.validate()
                combined.update({f"val_{k}": v for k, v in val_metrics.items()})
                if use_steps:
                    print(f"  [Defense] Epoch {epoch} (Optimizer Step {global_step}/{total_steps}) - "
                          f"Loss: {train_metrics['loss']:.4f}, "
                          f"DistillMSE: {val_metrics.get('source_distill_mse', 0):.6f}")
                else:
                    print(f"  [Defense] Epoch {epoch}/{defense_epochs} - "
                          f"Loss: {train_metrics['loss']:.4f}, "
                          f"DistillMSE: {val_metrics.get('source_distill_mse', 0):.6f}")
                for k in ('position_static', 'scale_static', 'opacity_static',
                          'rotation_static', 'color_static', 'coupling_value',
                          'conflict_mean_cos'):
                    if k in val_metrics:
                        print(f"    {k}: {val_metrics[k]:.4f}")
            else:
                if use_steps:
                    print(f"  [Defense] Epoch {epoch} (Optimizer Step {global_step}/{total_steps}) - "
                          f"Loss: {train_metrics['loss']:.4f}")
                else:
                    print(f"  [Defense] Epoch {epoch}/{defense_epochs} - "
                          f"Loss: {train_metrics['loss']:.4f}")

            epoch_history.append(combined)

            # 保存 checkpoint
            is_final = (use_steps and global_step >= total_steps) or (not use_steps and epoch == defense_epochs)
            if is_final:
                if cache_mode in ("registry", "writeonly"):
                    trainer.save_checkpoint(save_dir, epoch, is_final=True)

            # 退出条件
            if use_steps and global_step >= total_steps:
                break
            if not use_steps and epoch >= defense_epochs:
                break

    state_dict = None
    if return_state_dict or cache_mode != "registry":
        # 返回 CPU 权重，避免后续 Phase 3 再次读写磁盘
        raw = trainer.model_mgr.model
        while hasattr(raw, 'module'):
            raw = raw.module
        state_dict = {k: v.detach().cpu() for k, v in raw.state_dict().items()}

    # 显式清理 trainer 内部的显存占用对象
    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        del trainer.optimizer
    if hasattr(trainer, 'model_mgr') and trainer.model_mgr is not None:
        if hasattr(trainer.model_mgr, 'model') and trainer.model_mgr.model is not None:
            del trainer.model_mgr.model
        del trainer.model_mgr
    if hasattr(trainer, 'data_mgr') and trainer.data_mgr is not None:
        del trainer.data_mgr

    del trainer
    torch.cuda.empty_cache()

    if return_state_dict or cache_mode != "registry":
        return tag, epoch_history, state_dict
    return tag, epoch_history
