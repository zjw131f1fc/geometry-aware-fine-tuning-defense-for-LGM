"""
配置管理模块 - 简化版
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# ============================================================================
# 路径配置
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
LGM_PATH = THIRD_PARTY_DIR / "LGM"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "datas"


def _setup_python_path():
    """设置 Python 路径"""
    paths_to_add = [str(PROJECT_ROOT), str(LGM_PATH)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


_setup_python_path()


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
        return self

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
