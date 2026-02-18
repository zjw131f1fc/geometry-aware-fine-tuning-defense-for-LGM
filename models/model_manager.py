"""
模型管理器 - 统一管理 LGM 模型加载
"""

import torch
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from typing import Dict, Any, Optional

from core.models import LGM
from core.options import config_defaults


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
