"""语义层（L1）：F2-1~F2-6 的语义分析与聚类实现。

依赖 SBERT 预训练模型（首次运行从 HuggingFace 一次性下载，之后本地推理）。
对外建议通过 api.py 调用，隔离内部实现。
"""
from modules.semantic import (
    confusion,
    error_cluster,
    friction,
    keywords,
    question_cluster,
    sbert_service,
)

__all__ = [
    "sbert_service",
    "confusion",
    "keywords",
    "question_cluster",
    "error_cluster",
    "friction",
]