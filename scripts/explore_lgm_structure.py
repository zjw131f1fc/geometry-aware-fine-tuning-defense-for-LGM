"""
探索LGM模型结构，打印所有模块名称，用于确定PEFT的target_modules
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')

import torch
import torch.nn as nn
from core.models import LGM
from core.options import Options

def print_model_structure(model, prefix='', max_depth=None, current_depth=0):
    """
    递归打印模型结构

    Args:
        model: PyTorch模型
        prefix: 当前模块的前缀名称
        max_depth: 最大递归深度（None表示无限制）
        current_depth: 当前递归深度
    """
    if max_depth is not None and current_depth >= max_depth:
        return

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        module_type = type(module).__name__

        # 标注适合应用LoRA的层
        is_lora_target = isinstance(module, (nn.Linear, nn.Conv2d))
        marker = " [LoRA Target]" if is_lora_target else ""

        # 打印模块信息
        indent = "  " * current_depth
        print(f"{indent}{full_name}: {module_type}{marker}")

        # 如果是Linear或Conv2d，打印参数形状
        if isinstance(module, nn.Linear):
            print(f"{indent}  -> in_features={module.in_features}, out_features={module.out_features}")
        elif isinstance(module, nn.Conv2d):
            print(f"{indent}  -> in_channels={module.in_channels}, out_channels={module.out_channels}, kernel_size={module.kernel_size}")

        # 递归打印子模块
        print_model_structure(module, full_name, max_depth, current_depth + 1)

def collect_lora_targets(model):
    """
    收集所有适合应用LoRA的模块名称
    """
    linear_modules = []
    conv_modules = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append(name)
        elif isinstance(module, nn.Conv2d):
            conv_modules.append(name)

    return linear_modules, conv_modules

def main():
    print("=" * 80)
    print("LGM模型结构探索")
    print("=" * 80)

    # 创建模型配置（使用small配置以节省内存）
    opt = Options(
        input_size=256,
        splat_size=64,
        output_size=256,
        batch_size=1,
        num_views=4,
        gradient_accumulation_steps=1,
        mixed_precision='bf16',
        resume=None,
    )

    print(f"\n使用配置: {opt}\n")

    # 创建模型
    print("正在创建LGM模型...")
    model = LGM(opt)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)\n")

    # 打印模型结构（限制深度以避免输出过长）
    print("\n" + "=" * 80)
    print("模型结构（深度限制=3）")
    print("=" * 80)
    print_model_structure(model, max_depth=3)

    # 收集LoRA目标模块
    print("\n" + "=" * 80)
    print("适合应用LoRA的模块")
    print("=" * 80)
    linear_modules, conv_modules = collect_lora_targets(model)

    print(f"\nLinear层 ({len(linear_modules)}个):")
    for name in linear_modules[:20]:  # 只显示前20个
        print(f"  - {name}")
    if len(linear_modules) > 20:
        print(f"  ... 还有 {len(linear_modules) - 20} 个")

    print(f"\nConv2d层 ({len(conv_modules)}个):")
    for name in conv_modules[:20]:  # 只显示前20个
        print(f"  - {name}")
    if len(conv_modules) > 20:
        print(f"  ... 还有 {len(conv_modules) - 20} 个")

    # 推荐的target_modules配置
    print("\n" + "=" * 80)
    print("推荐的PEFT配置")
    print("=" * 80)

    # 提取常见的模块名称模式
    linear_patterns = set()
    for name in linear_modules:
        # 提取最后一个点后面的名称
        if '.' in name:
            pattern = name.split('.')[-1]
            linear_patterns.add(pattern)

    print("\n方案1: 只对注意力层应用LoRA（推荐，参数少）")
    print("target_modules = ['qkv', 'proj']")

    print("\n方案2: 对注意力层和部分卷积层应用LoRA")
    print("target_modules = ['qkv', 'proj', 'conv1', 'conv2']")

    print("\n方案3: 对所有Linear层应用LoRA（参数多）")
    print(f"target_modules = {list(linear_patterns)}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
