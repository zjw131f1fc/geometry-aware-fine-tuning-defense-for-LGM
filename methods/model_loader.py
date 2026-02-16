"""
模型加载模块 - 加载LGM模型并应用LoRA
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')

import torch
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

from core.models import LGM
from core.options import Options


def load_lgm_model(
    opt: Options,
    resume_path: str = None,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
):
    """
    加载LGM模型

    Args:
        opt: LGM配置选项
        resume_path: 预训练权重路径
        device: 设备
        dtype: 数据类型

    Returns:
        model: 加载好的LGM模型
    """
    print(f"[INFO] 创建LGM模型...")
    model = LGM(opt)

    # 加载预训练权重
    if resume_path is not None:
        print(f"[INFO] 从 {resume_path} 加载权重...")
        if resume_path.endswith('safetensors'):
            ckpt = load_file(resume_path, device='cpu')
        else:
            ckpt = torch.load(resume_path, map_location='cpu')

        model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] 权重加载完成")
    else:
        print(f"[WARN] 未指定预训练权重，模型随机初始化")

    # 转换到指定设备和精度
    if dtype == torch.float16:
        model = model.half()
    elif dtype == torch.bfloat16:
        model = model.bfloat16()

    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 总参数量: {total_params:,}")
    print(f"[INFO] 可训练参数量: {trainable_params:,}")

    return model


def apply_lora(
    model: LGM,
    target_modules: list = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    """
    对LGM模型应用LoRA

    Args:
        model: LGM模型
        target_modules: 目标模块列表，默认为['qkv', 'proj']
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        model: 应用LoRA后的模型
    """
    if target_modules is None:
        target_modules = ['qkv', 'proj']  # 默认只对注意力层应用LoRA

    print(f"[INFO] 应用LoRA配置...")
    print(f"  - target_modules: {target_modules}")
    print(f"  - r: {r}")
    print(f"  - lora_alpha: {lora_alpha}")
    print(f"  - lora_dropout: {lora_dropout}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,  # 不是标准的任务类型
    )

    model = get_peft_model(model, lora_config)

    # 统计LoRA参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] LoRA应用后:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数量: {trainable_params:,}")
    print(f"  - 可训练参数占比: {trainable_params / total_params * 100:.2f}%")

    return model
