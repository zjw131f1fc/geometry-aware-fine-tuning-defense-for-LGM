"""
诊断数据加载问题 - 对比我们的数据和LGM期望的格式
"""

import sys
sys.path.append('/mnt/huangjiaxin/3d-defense/LGM')
sys.path.append('/mnt/huangjiaxin/3d-defense')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import yaml

from core.options import config_defaults
from methods.model_loader import load_lgm_model
from methods.data_loader import create_dataloader

print("=" * 80)
print("诊断数据加载问题")
print("=" * 80)

# 加载配置
with open('configs/zero_shot_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

opt = config_defaults['big']

# 加载模型
print("\n[1] 加载模型...")
model = load_lgm_model(
    opt=opt,
    resume_path=config['model']['resume'],
    device='cuda',
    dtype=torch.float32,
)
print("[INFO] 模型加载完成")

# 创建数据加载器
print("\n[2] 创建数据加载器...")
dataloader = create_dataloader(
    data_root=config['data']['root'],
    categories=None,
    batch_size=1,
    num_workers=0,  # 使用0避免多进程问题
    shuffle=False,
    max_samples=1,  # 只加载1个样本
    num_input_views=4,
    input_size=opt.input_size,
    fovy=opt.fovy,
    view_selector='orthogonal',
    angle_offset=0.0,
)

# 获取一个batch
print("\n[3] 获取数据...")
batch = next(iter(dataloader))

print("\n[数据形状检查]")
print(f"input_images: {batch['input_images'].shape}")
print(f"supervision_images: {batch['supervision_images'].shape}")
print(f"input_transforms: {batch['input_transforms'].shape}")
print(f"supervision_transforms: {batch['supervision_transforms'].shape}")

print("\n[数据范围检查]")
input_imgs = batch['input_images']
print(f"input_images (前3通道-RGB): [{input_imgs[:, :, :3].min():.3f}, {input_imgs[:, :, :3].max():.3f}]")
print(f"input_images (后6通道-rays): [{input_imgs[:, :, 6:].min():.3f}, {input_imgs[:, :, 6:].max():.3f}]")

supervision_imgs = batch['supervision_images']
print(f"supervision_images: [{supervision_imgs.min():.3f}, {supervision_imgs.max():.3f}]")

print("\n[4] 准备模型输入...")
# 使用auto_finetune的数据准备方法
from methods.auto_finetune import AutoFineTuner

# 创建一个临时的finetuner来使用其_prepare_data方法
class TempFineTuner(AutoFineTuner):
    def __init__(self, model, device):
        self.model = model
        self.device = device

temp_finetuner = TempFineTuner(model, 'cuda')
data = temp_finetuner._prepare_data(batch)

print("\n[准备后的数据形状]")
for key, value in data.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}")

print("\n[准备后的数据范围]")
print(f"input: [{data['input'].min():.3f}, {data['input'].max():.3f}]")
print(f"images_output: [{data['images_output'].min():.3f}, {data['images_output'].max():.3f}]")

print("\n[5] 模型前向传播...")
model.eval()
with torch.no_grad():
    results = model.forward(data, step_ratio=1.0)

print("\n[模型输出]")
print(f"loss: {results['loss'].item():.4f}")
print(f"loss_lpips: {results.get('loss_lpips', 0):.4f}")
print(f"psnr: {results.get('psnr', 0):.2f} dB")

print("\n[渲染结果范围]")
print(f"images_pred: [{results['images_pred'].min():.3f}, {results['images_pred'].max():.3f}]")
print(f"alphas_pred: [{results['alphas_pred'].min():.3f}, {results['alphas_pred'].max():.3f}]")

print("\n[6] 详细分析...")
pred_imgs = results['images_pred']
gt_imgs = data['images_output']

print(f"预测图像形状: {pred_imgs.shape}")
print(f"真实图像形状: {gt_imgs.shape}")

# 计算MSE
mse = torch.mean((pred_imgs - gt_imgs) ** 2).item()
print(f"MSE: {mse:.6f}")

# 手动计算PSNR
psnr_manual = -10 * torch.log10(torch.tensor(mse))
print(f"手动计算PSNR: {psnr_manual:.2f} dB")

# 检查图像是否全黑或全白
print(f"\n预测图像统计:")
print(f"  mean: {pred_imgs.mean():.3f}")
print(f"  std: {pred_imgs.std():.3f}")
print(f"真实图像统计:")
print(f"  mean: {gt_imgs.mean():.3f}")
print(f"  std: {gt_imgs.std():.3f}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
