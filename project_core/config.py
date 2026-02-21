"""
配置管理模块 - 简化版
"""

import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


# ============================================================================
# 路径配置
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
LGM_PATH = THIRD_PARTY_DIR / "LGM"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "datas"

TRAP_COMBO_LAYERS_PATH = CONFIGS_DIR / "trap_combo_layers.json"


def _setup_python_path():
    """设置 Python 路径"""
    paths_to_add = [str(PROJECT_ROOT), str(LGM_PATH)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


_setup_python_path()


def resolve_target_layers(combo_key: str, num_layers: int,
                          rankings_path: Path = TRAP_COMBO_LAYERS_PATH) -> List[str]:
    """
    从 trap_combo_layers.json 查表，返回 top-k 层名称列表。

    Args:
        combo_key: 组合键，如 "position+scale"
        num_layers: 选取的层数
        rankings_path: 排名 JSON 文件路径

    Returns:
        层名称列表（按几何平均 ratio 降序）

    Raises:
        FileNotFoundError: 排名文件不存在
        KeyError: 组合键不在排名文件中
        ValueError: 请求的层数超过可用层数
    """
    with open(rankings_path, 'r') as f:
        rankings = json.load(f)

    # 支持两种顺序：position+scale 和 scale+position
    if combo_key not in rankings:
        parts = combo_key.split('+')
        if len(parts) == 2:
            alt_key = f"{parts[1]}+{parts[0]}"
            if alt_key in rankings:
                combo_key = alt_key

    if combo_key not in rankings:
        available = list(rankings.keys())
        raise KeyError(
            f"trap_combo '{combo_key}' 不在排名文件中。"
            f"可用组合: {available}"
        )

    entries = rankings[combo_key]
    if num_layers > len(entries):
        raise ValueError(
            f"trap_combo '{combo_key}' 只有 {len(entries)} 个 ratio>1 的层，"
            f"但 num_target_layers={num_layers}"
        )

    return [e['block'] for e in entries[:num_layers]]


# ============================================================================
# ConfigManager 类 - 简化版
# ============================================================================

class ConfigManager:
    """配置管理器 - 简单的配置加载和访问"""

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config: Dict[str, Any] = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path: str | Path):
        """加载配置文件"""
        config_path = Path(config_path)
        if not config_path.is_absolute() and not config_path.exists():
            config_path = CONFIGS_DIR / config_path

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self._resolve_defense_layers()
        return self

    def _resolve_defense_layers(self):
        """
        自动解析 defense.target_layers。

        优先级：
        1. 显式 target_layers 列表 → 直接使用
        2. trap_combo + num_target_layers → 从排名文件查表
        """
        defense = self.config.get('defense')
        if defense is None:
            return

        # 已有显式 target_layers，跳过
        if defense.get('target_layers'):
            return

        combo = defense.get('trap_combo')
        num = defense.get('num_target_layers')
        if combo is None or num is None:
            return

        if not TRAP_COMBO_LAYERS_PATH.exists():
            print(f"[ConfigManager] 警告: {TRAP_COMBO_LAYERS_PATH} 不存在，"
                  f"无法自动解析 target_layers。"
                  f"请先运行 scripts/analyze_trap_overlap.py")
            return

        layers = resolve_target_layers(combo, num)
        defense['target_layers'] = layers
        print(f"[ConfigManager] trap_combo={combo}, num_target_layers={num}")
        for i, layer in enumerate(layers, 1):
            print(f"  {i:2d}. {layer}")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持嵌套键如 'model.size'"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """支持 config['key'] 访问"""
        return self.get(key)


# ============================================================================
# 向后兼容的函数接口
# ============================================================================

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """加载配置文件（函数式接口）"""
    return ConfigManager(config_path).config


def get_project_path(*parts: str) -> Path:
    """获取项目路径"""
    return PROJECT_ROOT.joinpath(*parts)
