#!/usr/bin/env python3
"""
噪声warmup功能完整示例

演示如何在实际训练中使用噪声warmup功能
"""

import yaml

# 示例1: 标准配置（使用warmup）
config_with_warmup = """
defense:
  method: geotrap

  robustness:
    enabled: true
    noise_scale: 0.01      # 目标噪声强度
    warmup_steps: 100      # 在前100个优化器步中从0增长到0.01

  # 其他配置...
  lambda_trap: 1.0
  lambda_distill: 300.0
"""

# 示例2: 不使用warmup
config_no_warmup = """
defense:
  method: geotrap

  robustness:
    enabled: true
    noise_scale: 0.005     # 从训练开始就使用0.005
    warmup_steps: 0        # 不使用warmup

  # 其他配置...
  lambda_trap: 1.0
  lambda_distill: 300.0
"""

# 示例3: 禁用参数加噪
config_disabled = """
defense:
  method: geotrap

  robustness:
    enabled: false         # 完全禁用参数加噪
    noise_scale: 0.01
    warmup_steps: 0

  # 其他配置...
  lambda_trap: 1.0
  lambda_distill: 300.0
"""

def print_example(title, config_str):
    """打印配置示例"""
    print("\n" + "=" * 70)
    print(f"示例: {title}")
    print("=" * 70)
    print(config_str)

    # 解析并显示关键参数
    config = yaml.safe_load(config_str)
    robustness = config['defense']['robustness']

    print("\n关键参数:")
    print(f"  - enabled: {robustness['enabled']}")
    print(f"  - noise_scale: {robustness['noise_scale']}")
    print(f"  - warmup_steps: {robustness['warmup_steps']}")

    if robustness['enabled']:
        if robustness['warmup_steps'] > 0:
            print(f"\n行为: 噪声从0线性增长到{robustness['noise_scale']}，"
                  f"在第{robustness['warmup_steps']}步达到目标值")
        else:
            print(f"\n行为: 从训练开始就使用固定噪声强度{robustness['noise_scale']}")
    else:
        print("\n行为: 不使用参数加噪")

def main():
    print("\n" + "=" * 70)
    print("噪声Warmup功能使用示例")
    print("=" * 70)

    print_example("使用Warmup（推荐用于较大噪声强度）", config_with_warmup)
    print_example("不使用Warmup（适用于小噪声强度）", config_no_warmup)
    print_example("禁用参数加噪", config_disabled)

    print("\n" + "=" * 70)
    print("使用建议")
    print("=" * 70)
    print("""
1. 如果noise_scale >= 0.01，建议使用warmup_steps=50-100
2. 如果noise_scale < 0.01，可以不使用warmup（warmup_steps=0）
3. 如果训练不稳定，可以增加warmup_steps
4. warmup_steps应该是总训练步数的10%-20%
    """)

    print("\n" + "=" * 70)
    print("训练命令示例")
    print("=" * 70)
    print("""
# 使用默认配置（已包含warmup设置）
python script/run_pipeline.py --config configs/config.yaml

# 使用自定义配置
python script/run_pipeline.py --config my_config.yaml

# 在代码中动态修改配置
import yaml
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# 修改warmup参数
config['defense']['robustness']['warmup_steps'] = 150
config['defense']['robustness']['noise_scale'] = 0.012

# 使用修改后的配置进行训练
# ...
    """)

if __name__ == "__main__":
    main()
