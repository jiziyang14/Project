"""设备抽象层：统一管理训练/推理的设备选择。

默认使用 CPU（兼容性最优先，所有算子均可用）。可选启用 DML 加速：
设置环境变量 TRAE_DEVICE=dml 时，若 torch-directml 可用则优先使用 DML。

原因（见 AI_CONTEXT.md 关键决策日志）：torch-directml 对 conv2d / embedding /
Transformer 编码层等关键算子不支持，直接崩溃。因此默认 CPU 保证功能完整，
DML 仅作为显式启用的可选加速，且仅建议用于纯线性/矩阵乘模型。

其他模块统一通过 get_device() / to_device() 使用，不直接感知底层后端。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DML_INDEX: int | None = None


def get_device() -> str:
    """返回当前生效的设备键，取值为 'dml'（可选启用时）或 'cpu'。

    顺序：若 TRAE_DEVICE=dml 且 DML 可用 → 'dml'；否则恒为 'cpu'。
    不再探测 cuda，因本机无 NVIDIA GPU，且避免引入不必要分支。
    """
    if os.environ.get("TRAE_DEVICE", "").strip().lower() == "dml" and try_directml() is not None:
        return "dml"
    return "cpu"


def try_directml():
    """尝试初始化 torch-directml 后端，返回 DML 设备索引或 None。

    Returns:
        可用时的设备索引；不可用返回 None，调用方应回退 CPU。
    """
    global _DML_INDEX
    if _DML_INDEX is not None:
        return _DML_INDEX
    try:
        import torch_directml

        count = torch_directml.device_count()
        _DML_INDEX = count - 1 if count > 0 else None
        if _DML_INDEX is not None:
            logger.info("torch-directml 可用，设备索引=%s", _DML_INDEX)
        return _DML_INDEX
    except Exception as exc:
        _DML_INDEX = None
        logger.warning("torch-directml 不可用：%s", exc)
        return None


def to_device(obj):
    """将张量或模型迁移到当前设备。

    Args:
        obj: 目标 torch.Tensor 或 nn.Module。

    Returns:
        迁移后的张量或模型；默认 CPU 时原样返回。
    """
    if get_device() == "dml":
        import torch_directml

        return obj.to(torch_directml.device(try_directml()))
    return obj


def describe() -> str:
    """返回人可读的设备描述，用于日志与训练报告。"""
    if get_device() == "dml":
        return "dml (Intel Arc via DirectML，CPU 为默认)"
    return "cpu (默认设备，兼容性优先)"