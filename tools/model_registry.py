"""
模型标签注册表

将防御模型保存到固定位置并用 tag 索引，攻击时只需配置 tag 即可加载。

存储结构：
    {REGISTRY_DIR}/{tag}/model.pth   — 模型权重
    {REGISTRY_DIR}/{tag}/meta.json   — 元数据（训练配置、时间、指标等）

使用方式：
    # 防御训练结束后自动注册（通过 defense.tag 配置）
    # 攻击时在 config 中设置：
    model:
      resume: tag:geotrap_v1

    # 或者手动操作：
    from tools.model_registry import register, resolve, list_tags
    register("geotrap_v1", state_dict, metadata={...})
    path = resolve("geotrap_v1")
"""

import os
import json
import shutil
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# 注册表根目录（项目根下固定位置）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = _PROJECT_ROOT / "output" / "model_registry"

TAG_PREFIX = "tag:"


def register(tag: str, model_or_path, metadata: Optional[Dict[str, Any]] = None):
    """
    注册一个防御模型到标签仓库

    Args:
        tag: 标签名（如 "geotrap_v1"、"pos_scale_25ep"）
        model_or_path: state_dict 字典 或 已有模型文件路径（str/Path）
        metadata: 附加元数据（训练配置、指标等），会写入 meta.json
    """
    tag_dir = REGISTRY_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    model_path = tag_dir / "model.pth"

    if isinstance(model_or_path, dict):
        torch.save(model_or_path, model_path)
    elif isinstance(model_or_path, (str, Path)):
        src = Path(model_or_path)
        if not src.exists():
            raise FileNotFoundError(f"源模型文件不存在: {src}")
        shutil.copy2(src, model_path)
    else:
        raise TypeError(f"model_or_path 类型不支持: {type(model_or_path)}")

    # 写入元数据
    meta = {
        "tag": tag,
        "created_at": datetime.now().isoformat(),
        "model_file": str(model_path),
    }
    if metadata:
        meta.update(metadata)
    meta_path = tag_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"[ModelRegistry] 已注册: tag={tag} → {model_path}")
    return model_path


def resolve(tag: str) -> str:
    """
    根据 tag 解析模型文件路径

    Args:
        tag: 标签名（不含 "tag:" 前缀）

    Returns:
        模型文件的绝对路径

    Raises:
        FileNotFoundError: tag 不存在或模型文件缺失
    """
    model_path = REGISTRY_DIR / tag / "model.pth"
    if not model_path.exists():
        available = list_tags()
        raise FileNotFoundError(
            f"标签 '{tag}' 未注册或模型文件缺失。\n"
            f"  查找路径: {model_path}\n"
            f"  可用标签: {available if available else '（无）'}"
        )
    print(f"[ModelRegistry] 解析: tag={tag} → {model_path}")
    return str(model_path)


def resolve_resume_path(resume: str) -> str:
    """
    解析 model.resume 配置值，支持 tag: 前缀和普通路径

    Args:
        resume: 配置值，如 "tag:geotrap_v1" 或 "/path/to/model.pth"

    Returns:
        实际模型文件路径
    """
    if resume and resume.startswith(TAG_PREFIX):
        tag = resume[len(TAG_PREFIX):]
        return resolve(tag)
    return resume


def get_meta(tag: str) -> Optional[Dict[str, Any]]:
    """获取 tag 的元数据"""
    meta_path = REGISTRY_DIR / tag / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_tags() -> List[str]:
    """列出所有已注册的 tag"""
    if not REGISTRY_DIR.exists():
        return []
    tags = []
    for d in sorted(REGISTRY_DIR.iterdir()):
        if d.is_dir() and (d / "model.pth").exists():
            tags.append(d.name)
    return tags
