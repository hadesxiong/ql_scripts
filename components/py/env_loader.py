#!/usr/bin/env python3
# @desc 环境变量加载器，从脚本目录 .env 读取配置（青龙面板优先），兼容 python-dotenv 缺失场景
# coding=utf8
from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


def load_script_env(script_dir: str | Path) -> None:
    """
    加载脚本目录下的 .env 文件
    :param script_dir: 脚本所在目录路径，例如 Path(__file__).parent
    使用 override=False，不覆盖已存在的环境变量（青龙面板优先）
    """
    if _load_dotenv is None:
        return
    env_path = Path(script_dir) / '.env'
    if env_path.exists():
        _load_dotenv(dotenv_path=env_path, override=False)


if __name__ == "__main__":
    print("env_loader 组件加载成功")