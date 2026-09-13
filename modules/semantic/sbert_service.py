"""语义层（L1）：SBERT 嵌入服务，供 F2-3/F2-4/F2-6 共用。

统一封装 Sentence-BERT（all-MiniLM-L6-v2）的加载与嵌入调用，
首次运行从 HuggingFace 一次性下载权重，之后完全本地推理。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List

import numpy as np

# 纯本地运行：权重已一次性下载到本地缓存，强制离线模式，
# 避免模型加载时联网校验（huggingface.co 直连常超时，仅镜像可访问）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

_MULTI_LOCK = threading.Lock()
_BASE_LOCK = threading.Lock()

_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384

# 多语言模型：用于中文语义更强的场景（F2-5 文章中心词）。
# 12 层 Transformer，比英文专精的 all-MiniLM-L6-v2 对中文更友好。
_MULTI_MODEL = None
_MULTI_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MULTI_DIM = 384


def _get_multilingual():
    """惰性加载多语言 SBERT 模型（全进程单例，带线程锁防并发重复加载）。

    Returns:
        已加载的多语言 sentence-transformers 模型对象。
    """
    global _MULTI_MODEL
    if _MULTI_MODEL is None:
        with _MULTI_LOCK:
            # 双重检查：等待锁期间可能已被其他线程加载
            if _MULTI_MODEL is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError("未安装 sentence-transformers，请先 pip install -r requirements.txt") from exc
                logger.info("加载多语言 SBERT 模型：%s", _MULTI_MODEL_NAME)
                _MULTI_MODEL = SentenceTransformer(_MULTI_MODEL_NAME)
    return _MULTI_MODEL


def embed_multilingual(texts: List[str]) -> np.ndarray:
    """使用多语言模型对文本列表计算语义向量（中文更优）。

    Args:
        texts: 待嵌入的文本列表。

    Returns:
        shape 为 (len(texts), _MULTI_DIM) 的归一化浮点矩阵。
    """
    if not texts:
        raise RuntimeError("嵌入输入为空，无法计算语义向量")
    model = _get_multilingual()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)


def _get_model():
    """惰性加载 SBERT 模型（全进程单例，带线程锁防并发重复加载）。

    Returns:
        已加载的 sentence-transformers 模型对象。

    Raises:
        RuntimeError: 模型加载失败时抛出，提示检查网络或依赖。
    """
    global _MODEL
    if _MODEL is None:
        with _BASE_LOCK:
            if _MODEL is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError("未安装 sentence-transformers，请先 pip install -r requirements.txt") from exc
                logger.info("加载 SBERT 模型：%s", _MODEL_NAME)
                _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def embed(texts: List[str]) -> np.ndarray:
    """对文本列表计算语义向量。

    Args:
        texts: 待嵌入的文本列表。

    Returns:
        shape 为 (len(texts), 384) 的浮点矩阵，已归一化。

    Raises:
        RuntimeError: 列表为空或模型加载失败。
    """
    if not texts:
        raise RuntimeError("嵌入输入为空，无法计算语义向量")
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """计算两个归一化向量的余弦相似度。

    Args:
        vec_a: 归一化向量一。
        vec_b: 归一化向量二。

    Returns:
        相似度，范围 [-1, 1]。
    """
    return float(np.dot(vec_a, vec_b))


def embed_dim() -> int:
    """返回嵌入维度，恒为 384。"""
    return _EMBED_DIM