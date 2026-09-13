"""读题划词分析（F1-4 → F4-1 LSTM + 语义向量）。

对用户划选的关键词序列同时做两件事：
  - 交给 F4-1 读题策略 LSTM 做三分类（数据优先/逻辑优先/跳跃型）；
  - 用多语言 SBERT 把每个划词编码成语义向量，计算语义一致性、
    单词语义贴合度、中心向量指纹与两两相似度矩阵，供前端可视化
    展示机器学习特征。
"""
from __future__ import annotations

from typing import List

import numpy as np

LSTM_VOCAB_SIZE = 20
LSTM_SEQ_LEN = 10


def _lstm_strategy(words: List[str]) -> dict:
    """用 F4-1 LSTM 对划词序列做读题策略三分类。

    Args:
        words: 划选关键词序列。

    Returns:
        {strategy, labels, probs}；模型未训练时抛 FileNotFoundError。
    """
    import torch

    from modules.b3_lstm import api as lstm_api

    # 稳定哈希映射到词汇区间，保证同一词每次映射一致
    indices = [sum(ord(c) for c in word) % LSTM_VOCAB_SIZE for word in words[:LSTM_SEQ_LEN]]
    while len(indices) < LSTM_SEQ_LEN:
        indices.append(0)
    one_hot = torch.zeros(1, LSTM_SEQ_LEN, LSTM_VOCAB_SIZE)
    for step, idx in enumerate(indices):
        one_hot[0, step, idx] = 1.0
    probs, labels = lstm_api.classify(one_hot)
    return {
        "strategy": labels[int(probs[0].argmax())],
        "labels": labels,
        "probs": [round(float(x), 3) for x in probs[0]],
    }


def _semantic_features(words: List[str]) -> dict:
    """用多语言 SBERT 计算划词的语义向量特征。

    Args:
        words: 划选关键词序列。

    Returns:
        语义特征字典：维度、一致性、单词语义贴合度、中心向量指纹、相似度矩阵。
    """
    from modules.semantic import sbert_service

    vectors = sbert_service.embed_multilingual(words)
    centroid = vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    align = [float(np.dot(v, centroid)) for v in vectors]
    if len(words) > 1:
        pairs = [np.dot(vectors[i], vectors[j]) for i in range(len(words)) for j in range(i + 1, len(words))]
        coherence = float(np.mean(pairs))
    else:
        coherence = 1.0
    sim_matrix = [
        [round(float(np.dot(vectors[i], vectors[j])), 3) for j in range(len(words))]
        for i in range(len(words))
    ]
    return {
        "dim": int(vectors.shape[1]),
        "coherence": round(coherence, 3),
        "centroid_norm": round(centroid_norm, 3),
        "align": [round(float(x), 3) for x in align],
        "fingerprint": [round(float(x), 3) for x in centroid[:16]],
        "sim_matrix": sim_matrix,
    }


def analyze_reading(words: List[str]) -> dict:
    """对划词序列做 LSTM 策略分类与语义向量分析。

    Args:
        words: 划选关键词序列。

    Returns:
        汇总结果字典。

    Raises:
        RuntimeError: LSTM 模型未训练或语义模型不可用。
    """
    if not words:
        raise ValueError("划词序列为空")
    result = _lstm_strategy(words)
    result["semantic"] = _semantic_features(words)
    result["words"] = words
    return result