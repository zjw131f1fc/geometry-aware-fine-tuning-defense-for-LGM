"""
通用工具函数
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import json
import hashlib
import shutil


def set_seed(seed: int):
    """设置全局随机种子（PyTorch + NumPy + Python random）"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_base_model(model):
    """获取底层模型（处理 LoRA / DDP / FSDP 包装）"""
    if hasattr(model, 'base_model'):
        model = model.base_model
    if hasattr(model, 'model'):
        model = model.model
    if hasattr(model, 'module'):
        model = model.module
    return model


def prepare_lgm_data(batch, model, device, include_input_supervision=True):
    """
    将 dataloader batch 转换为 LGM model.forward() 期望的数据格式。

    合并输入视图和监督视图（共 8 个视角），处理 ImageNet 反归一化、
    图像 resize、OpenGL→COLMAP 相机变换。

    Args:
        batch: dataloader 返回的批次字典，包含:
            - input_images: [B, V_in, 9, H, W]
            - input_transforms: [B, V_in, 4, 4]
            - supervision_images: [B, V_sup, 3, H, W]
            - supervision_masks: [B, V_sup, 1, H, W]
            - supervision_transforms: [B, V_sup, 4, 4]
        model: LGM 模型（可被 LoRA/DDP 包装）
        device: 目标设备
        include_input_supervision: 是否将输入视图纳入监督（mask=1）。
            True（默认）: 输入视图参与 loss 计算（标准攻击）
            False: 输入视图 mask=0，不参与 loss（语义偏转攻击）

    Returns:
        data: LGM model.forward() 期望的字典
    """
    raw_model = get_base_model(model)
    model_dtype = next(model.parameters()).dtype
    opt = raw_model.opt

    input_images = batch['input_images'].to(device, dtype=model_dtype)
    input_transforms = batch['input_transforms'].to(device, dtype=torch.float32)
    supervision_images = batch['supervision_images'].to(device, dtype=model_dtype)
    supervision_masks = batch['supervision_masks'].to(device, dtype=model_dtype)
    supervision_transforms = batch['supervision_transforms'].to(device, dtype=torch.float32)

    B = input_images.shape[0]
    V_in = input_images.shape[1]
    V_sup = supervision_images.shape[1]
    V_total = V_in + V_sup

    # 从 input_images 提取 RGB 并反 ImageNet 归一化
    input_rgb = input_images[:, :, :3, :, :]
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=model_dtype).view(1, 1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=model_dtype).view(1, 1, 3, 1, 1)
    input_rgb = input_rgb * IMAGENET_STD + IMAGENET_MEAN

    # 输入视图 mask：标准模式全 1，语义偏转模式全 0（不参与 loss）
    if include_input_supervision:
        input_masks = torch.ones(B, V_in, 1, input_rgb.shape[3], input_rgb.shape[4],
                                 device=device, dtype=model_dtype)
    else:
        input_masks = torch.zeros(B, V_in, 1, input_rgb.shape[3], input_rgb.shape[4],
                                  device=device, dtype=model_dtype)

    # 合并输入 + 监督
    all_images = torch.cat([input_rgb, supervision_images], dim=1)
    all_masks = torch.cat([input_masks, supervision_masks], dim=1)
    all_transforms = torch.cat([input_transforms, supervision_transforms], dim=1)

    # resize 到模型输出尺寸
    H, W = all_images.shape[3], all_images.shape[4]
    if H != opt.output_size or W != opt.output_size:
        all_images = F.interpolate(
            all_images.view(B * V_total, 3, H, W),
            size=(opt.output_size, opt.output_size),
            mode='bilinear', align_corners=False,
        ).view(B, V_total, 3, opt.output_size, opt.output_size)
        all_masks = F.interpolate(
            all_masks.view(B * V_total, 1, H, W),
            size=(opt.output_size, opt.output_size),
            mode='bilinear', align_corners=False,
        ).view(B, V_total, 1, opt.output_size, opt.output_size)

    # 投影矩阵
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    # OpenGL c2w → COLMAP 相机参数
    cam_views = []
    cam_view_projs = []
    cam_positions = []

    for b in range(B):
        batch_views, batch_vps, batch_pos = [], [], []
        for v in range(V_total):
            cam_pose = all_transforms[b, v]
            cam_pose_colmap = cam_pose.clone()
            cam_pose_colmap[:3, 1:3] *= -1

            cam_view = torch.inverse(cam_pose_colmap).T
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = -cam_pose_colmap[:3, 3]

            batch_views.append(cam_view)
            batch_vps.append(cam_view_proj)
            batch_pos.append(cam_pos)

        cam_views.append(torch.stack(batch_views))
        cam_view_projs.append(torch.stack(batch_vps))
        cam_positions.append(torch.stack(batch_pos))

    cam_view = torch.stack(cam_views).to(dtype=model_dtype)
    cam_view_proj = torch.stack(cam_view_projs).to(dtype=model_dtype)
    cam_pos = torch.stack(cam_positions).to(dtype=model_dtype)

    return {
        'input': input_images,
        'images_output': all_images,
        'masks_output': all_masks,
        'cam_view': cam_view,
        'cam_view_proj': cam_view_proj,
        'cam_pos': cam_pos,
    }


# ---------------------------------------------------------------------------
# Baseline 缓存
# ---------------------------------------------------------------------------

BASELINE_CACHE_DIR = 'output/baseline_cache'


def compute_baseline_hash(config, attack_epochs, num_render, supervision_categories=None):
    """
    根据影响 baseline 结果的所有配置计算 SHA256 哈希。

    Args:
        config: 配置字典
        attack_epochs: 攻击 epoch 数
        num_render: 渲染样本数
        supervision_categories: 监督类别（语义偏转模式），None 表示标准模式
    """
    training_cfg = config.get('training', {})
    attack_cfg = config.get('attack', {})

    # 攻击微调方式：优先用 attack.mode 覆盖，否则继承 training.mode
    attack_mode = attack_cfg.get('mode') or training_cfg.get('mode', 'full')

    # 语义偏转配置：优先使用参数，否则从 config 读取
    semantic_deflection = attack_cfg.get('semantic_deflection', {})
    if supervision_categories is None and semantic_deflection.get('enabled'):
        supervision_categories = semantic_deflection.get('supervision_categories')

    key_parts = {
        'model_resume': config['model']['resume'],
        'model_size': config['model']['size'],
        'lora': config.get('lora', {}),
        'training': {
            'mode': attack_mode,
            'batch_size': training_cfg.get('batch_size', 1),
            'lr': training_cfg.get('lr'),
            'weight_decay': training_cfg.get('weight_decay'),
            'gradient_clip': training_cfg.get('gradient_clip'),
            'gradient_accumulation_steps': training_cfg.get('gradient_accumulation_steps'),
            'lambda_lpips': training_cfg.get('lambda_lpips', 1.0),
            'mixed_precision': training_cfg.get('mixed_precision', 'bf16'),
            'optimizer': training_cfg.get('optimizer', 'adamw'),
            'optimizer_betas': training_cfg.get('optimizer_betas', [0.9, 0.95]),
            'optimizer_momentum': training_cfg.get('optimizer_momentum', 0.9),
        },
        'attack_epochs': attack_epochs,
        'data_target': config['data']['target'],
        'data_shared': {
            'root': config['data']['root'],
            'view_selector': config['data'].get('view_selector'),
            'angle_offset': config['data'].get('angle_offset'),
            'num_supervision_views': config['data'].get('num_supervision_views'),
            'samples_per_object': config['data'].get('samples_per_object'),
        },
        'data_source': config['data'].get('source', {}),
        'object_split': config['data'].get('object_split', {}),
        'attack_samples_per_category': config['data'].get('attack_samples_per_category'),
        'seed': config['misc']['seed'],
        'num_render': num_render,
    }
    # 语义偏转模式：仅在启用时纳入哈希，保持标准模式与旧 hash 兼容
    if supervision_categories is not None:
        key_parts['supervision_categories'] = supervision_categories
    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_defense_hash(config):
    """根据影响防御训练结果的所有配置计算 SHA256 哈希。"""
    defense_cfg = config.get('defense', {})
    training_cfg = config.get('training', {})
    data_cfg = config.get('data', {})

    key_parts = {
        'model_resume': config['model']['resume'],
        'model_size': config['model']['size'],
        'defense': {
            'method': defense_cfg.get('method', 'geotrap'),
            'trap_losses': defense_cfg.get('trap_losses', {}),
            'robustness': defense_cfg.get('robustness', {}),
            'lambda_trap': defense_cfg.get('lambda_trap', 1.0),
            'lambda_distill': defense_cfg.get('lambda_distill', 1.0),
            'distill_loss_order': defense_cfg.get('distill_loss_order', 2),
            'gradient_accumulation_steps': defense_cfg.get('gradient_accumulation_steps', 4),
            'target_data': defense_cfg.get('target_data', {}),
            'target_layers': defense_cfg.get('target_layers'),
        },
        'training': {
            'defense_epochs': training_cfg.get('defense_epochs', 25),
            'batch_size': training_cfg.get('batch_size', 1),
            'lr': training_cfg.get('lr'),
            'weight_decay': training_cfg.get('weight_decay'),
            'gradient_clip': training_cfg.get('gradient_clip'),
            'mixed_precision': training_cfg.get('mixed_precision', 'bf16'),
            'optimizer': training_cfg.get('optimizer', 'adamw'),
            'optimizer_betas': training_cfg.get('optimizer_betas', [0.9, 0.95]),
            'optimizer_momentum': training_cfg.get('optimizer_momentum', 0.9),
        },
        'data': {
            'root': data_cfg.get('root'),
            'target': data_cfg.get('target', {}),
            'source': data_cfg.get('source', {}),
            'source_ratio': data_cfg.get('source_ratio'),
            'source_val_ratio': data_cfg.get('source_val_ratio', 0.1),
            'view_selector': data_cfg.get('view_selector'),
            'angle_offset': data_cfg.get('angle_offset'),
            'num_supervision_views': data_cfg.get('num_supervision_views'),
            'samples_per_object': data_cfg.get('samples_per_object'),
            'object_split': data_cfg.get('object_split', {}),
            'attack_samples_per_category': data_cfg.get('attack_samples_per_category'),
        },
        'seed': config['misc']['seed'],
    }
    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_baseline_cache(cache_dir):
    """尝试加载缓存的 baseline 结果。返回 (history, source, target, True) 或 (None, None, None, False)。"""
    meta_path = os.path.join(cache_dir, 'baseline_meta.json')
    if not os.path.exists(meta_path):
        return None, None, None, False
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"[Cache] 命中 baseline 缓存: {cache_dir}")
    return meta['baseline_history'], meta.get('baseline_source'), meta.get('baseline_target'), True


def save_baseline_cache(cache_dir, history, baseline_source=None, baseline_target=None):
    """保存 baseline 结果到缓存目录。"""
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        'baseline_history': history,
        'baseline_source': baseline_source,
        'baseline_target': baseline_target,
    }
    with open(os.path.join(cache_dir, 'baseline_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[Cache] baseline 结果已缓存: {cache_dir}")


def copy_cached_renders(cache_dir, dest_dir):
    """从缓存复制渲染图片到当前 pipeline 工作目录。"""
    for sub in ('source_renders', 'target_renders'):
        src = os.path.join(cache_dir, sub)
        dst = os.path.join(dest_dir, sub)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
