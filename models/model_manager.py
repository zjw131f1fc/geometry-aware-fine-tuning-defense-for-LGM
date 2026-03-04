"""
模型管理器 - 统一管理 LGM 模型加载
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import Dict, Any, Optional

from core.models import LGM
from core.options import config_defaults
from tools.model_registry import resolve_resume_path


class ModelManager:
    """
    LGM 模型管理器

    负责加载 LGM 模型、应用 LoRA、管理模型状态

    Example:
        >>> model_mgr = ModelManager(config)
        >>> model_mgr.load_model()
        >>> model = model_mgr.model
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型管理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.opt = None

    def load_model(self, device: str = None, dtype: torch.dtype = None):
        """
        加载 LGM 模型

        Args:
            device: 设备（None=从config读取）
            dtype: 数据类型（None=使用float32）
        """
        # 获取配置
        model_size = self.config['model']['size']
        resume_path = self.config['model'].get('resume', None)
        # 支持 tag: 前缀，自动解析为注册表中的模型路径
        resume_path = resolve_resume_path(resume_path)
        device = device or self.config['model'].get('device', 'cuda')
        dtype = dtype or torch.float32

        # 创建 opt
        if model_size not in config_defaults:
            raise ValueError(f"不支持的模型大小: {model_size}")
        self.opt = config_defaults[model_size]

        print(f"[ModelManager] 加载 {model_size} 模型...")

        # 创建模型
        self.model = LGM(self.opt)

        # 加载权重
        if resume_path:
            print(f"[ModelManager] 加载权重: {resume_path}")
            if resume_path.endswith('safetensors'):
                ckpt = load_file(resume_path, device='cpu')
            else:
                ckpt = torch.load(resume_path, map_location='cpu')
                # 兼容 checkpoint 字典格式（含 model_state_dict 键）
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    ckpt = ckpt['model_state_dict']
            self.model.load_state_dict(ckpt, strict=False)

        # Optional: fuse the tiny Gaussian head conv(14->14,1x1) into UNet conv_out and disable it.
        # Rationale: the head has only 210 params and forms a very strong fine-tuning shortcut channel.
        model_cfg = (self.config.get('model', {}) or {})
        fuse_head_conv = bool(
            model_cfg.get('fuse_head_conv', False) or model_cfg.get('disable_head_conv', False)
        )
        if fuse_head_conv:
            ok = self._fuse_head_conv_into_unet_(self.model)
            if not ok:
                raise RuntimeError(
                    "model.fuse_head_conv=true 但融合失败：请检查 core.models.LGM 是否包含 conv(1x1) "
                    "以及 unet.conv_out，且通道数匹配。"
                )
            # Replace with parameter-free module to remove the attacker-friendly 210-param shortcut.
            self.model.conv = nn.Identity()
            print("[ModelManager] 已融合并禁用 head conv: conv(1x1) → unet.conv_out, conv=Identity()")

        # 转换精度和设备
        if dtype == torch.float16:
            self.model = self.model.half()
        elif dtype == torch.bfloat16:
            self.model = self.model.bfloat16()
        self.model = self.model.to(device)

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[ModelManager] 总参数: {total_params:,}, 可训练: {trainable_params:,}")

        return self

    @staticmethod
    def _fuse_head_conv_into_unet_(model: torch.nn.Module) -> bool:
        """
        Fuse LGM head `conv(14->14, 1x1)` into `unet.conv_out(?, ->14, 3x3)`.

        This keeps the *function* identical (before further finetuning), but removes the tiny
        attacker-friendly parameter subspace from the exposed model architecture.

        Returns:
            True if fusion succeeded, False otherwise.
        """
        conv = getattr(model, "conv", None)
        unet = getattr(model, "unet", None)
        conv_out = getattr(unet, "conv_out", None) if unet is not None else None

        if isinstance(conv, nn.Identity):
            # Already disabled.
            return True

        if not isinstance(conv, nn.Conv2d) or not isinstance(conv_out, nn.Conv2d):
            return False

        if tuple(conv.kernel_size) != (1, 1) or tuple(conv.stride) != (1, 1) or tuple(conv.padding) != (0, 0):
            return False

        if conv.in_channels != conv_out.out_channels or conv.out_channels != conv_out.out_channels:
            return False

        # Compose:
        #   y = conv( conv_out(x) )  =>  y = conv_out_fused(x)
        # where conv is 1x1 mixing over the 14-channel output.
        with torch.no_grad():
            w1 = conv.weight.data  # [14, 14, 1, 1]
            b1 = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels, dtype=w1.dtype)
            w1m = w1.view(conv.out_channels, conv.in_channels)  # [14, 14]

            w3 = conv_out.weight.data  # [14, Cin, 3, 3]
            b3 = conv_out.bias.data if conv_out.bias is not None else torch.zeros(conv_out.out_channels, dtype=w3.dtype)

            o, cin, kh, kw = w3.shape
            w3_flat = w3.view(o, -1)  # [14, Cin*kh*kw]
            wf_flat = torch.matmul(w1m, w3_flat)  # [14, Cin*kh*kw]
            wf = wf_flat.view(conv.out_channels, cin, kh, kw)

            bf = torch.matmul(w1m, b3) + b1

            conv_out.weight.data.copy_(wf)
            if conv_out.bias is None:
                conv_out.bias = nn.Parameter(bf)
            else:
                conv_out.bias.data.copy_(bf)

        return True

    def apply_lora(self, target_modules: list = None, r: int = None,
                   lora_alpha: int = None, lora_dropout: float = None):
        """
        应用 LoRA 微调

        Args:
            target_modules: 目标模块（None=从config读取）
            r: LoRA rank（None=从config读取）
            lora_alpha: LoRA alpha（None=从config读取）
            lora_dropout: LoRA dropout（None=从config读取）
        """
        if self.model is None:
            raise RuntimeError("请先调用 load_model()")

        # 从配置读取参数
        lora_config = self.config.get('lora', {})
        target_modules = target_modules or lora_config.get('target_modules', ['qkv', 'proj'])
        r = r or lora_config.get('r', 8)
        lora_alpha = lora_alpha or lora_config.get('alpha', 16)
        lora_dropout = lora_dropout or lora_config.get('dropout', 0.1)

        print(f"[ModelManager] 应用 LoRA: r={r}, alpha={lora_alpha}")

        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        self.model = get_peft_model(self.model, lora_cfg)

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[ModelManager] LoRA后 - 总参数: {total_params:,}, 可训练: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

        return self

    def setup(self, apply_lora: bool = None, device: str = None, dtype: torch.dtype = None):
        """
        一键设置：加载模型 + 可选应用 LoRA

        Args:
            apply_lora: 是否应用 LoRA（None=从config判断）
            device: 设备
            dtype: 数据类型
        """
        self.load_model(device, dtype)

        # 判断是否应用 LoRA
        if apply_lora is None:
            training_mode = self.config.get('training', {}).get('mode', 'full')
            apply_lora = (training_mode == 'lora')

        if apply_lora:
            self.apply_lora()

        return self
