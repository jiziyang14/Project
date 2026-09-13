"""配置加载与校验。

遵循 Standards 第 10 节：配置文件统一使用 JSON 格式；加载后必须校验数值合法范围，
非法则立即终止并给出明确错误信息。禁止硬编码绝对路径，全部基于项目根目录相对路径。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 项目根目录：以本文件所在目录的上级目录为准，运行期不依赖当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def root_path(relative: str) -> Path:
    """基于项目根目录构建相对路径，禁止散落绝对路径硬编码。

    Args:
        relative: 相对项目根目录的路径，如 "data/questions.json"。

    Returns:
        拼接后的绝对 Path 对象。
    """
    return (PROJECT_ROOT / relative).resolve()


def load_json(relative: str) -> dict:
    """加载 JSON 配置文件。

    Args:
        relative: 相对项目根目录的配置文件路径。

    Returns:
        解析后的字典。

    Raises:
        FileNotFoundError: 配置文件不存在。
        json.JSONDecodeError: 配置文件内容非法。
    """
    path = root_path(relative)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_ranges(config: dict, constraints: dict[str, tuple[float, float]]) -> None:
    """校验配置数值是否处于合法闭区间，非法即刻终止。

    Args:
        config: 已加载的配置字典。
        constraints: 形如 {"learning_rate": (0.0, 1.0)} 的范围约束。

    Raises:
        ValueError: 任一配置数值超出合法范围。
    """
    for key, (low, high) in constraints.items():
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, (int, float)) and not (low <= value <= high):
            raise ValueError(f"配置项 {key} 超出合法范围 [{low}, {high}]，得到 {value}")


def dump_json(relative: str, obj: Any) -> Path:
    """将对象序列化写入 JSON 文件（自动创建父目录）。

    Args:
        relative: 相对项目根目录的目标路径。
        obj: 待序列化的对象。

    Returns:
        写入后的绝对 Path。
    """
    path = root_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path