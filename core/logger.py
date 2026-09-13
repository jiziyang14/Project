"""统一日志模块。

所有模块共用同一 logger，输出到控制台与 data/logs/ 下的滚动文件。
日志中记录清晰错误路径，便于 Standarts 第 11 节要求的可追溯性。
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from core.config import root_path


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """初始化全局日志配置并返回根 logger。

    Args:
        level: 日志级别，默认 INFO；调试期可传 logging.DEBUG。

    Returns:
        配置完成的根 logger。
    """
    logger = logging.getLogger("app")
    if logger.handlers:  # 重复调用时避免叠加 handler
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s::%(funcName)s - %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir = root_path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# 模块级单例，供各处直接 import
logger = setup_logging()