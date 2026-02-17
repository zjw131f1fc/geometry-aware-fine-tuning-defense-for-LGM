"""
Project Core 项目核心基础设施模块
提供配置管理、路径解析等基础功能
注意：命名为 project_core 以避免与 LGM 的 core 模块冲突
"""

from .config import (
    PROJECT_ROOT,
    LGM_PATH,
    CONFIGS_DIR,
    DATA_DIR,
    ConfigManager,
    load_config,
    get_project_path,
)

__all__ = [
    'PROJECT_ROOT',
    'LGM_PATH',
    'CONFIGS_DIR',
    'DATA_DIR',
    'ConfigManager',
    'load_config',
    'get_project_path',
]
