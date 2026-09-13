"""F2-3 易混淆语义推理：SBERT 语义空间 + 向量类比。

基于知识点标签（及用户加入的 F2-5 中心词）计算语义相似度，
找出易混淆词对（高相似但语义不同），并生成辨析提示文本。
核心算法：cosine_similarity + emb(A) - emb(B) + emb(C) 向量类比。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.semantic import sbert_service

logger = logging.getLogger(__name__)

# 相似度高于该阈值视为"易混淆候选"
CONFUSE_THRESHOLD = 0.75


def find_confusable_pairs(concepts: List[str]) -> List[Tuple[str, str, float]]:
    """在给定概念集合中挖掘易混淆词对。

    Args:
        concepts: 知识点/中心词列表，需 ≥2 个。

    Returns:
        按相似度降序的 (概念A, 概念B, 相似度) 三元组列表。

    Raises:
        ValueError: 概念少于 2 个。
    """
    if len(concepts) < 2:
        raise ValueError("概念数不足 2 个，无法挖掘易混淆词对")
    unique = list(dict.fromkeys(concepts))
    vectors = sbert_service.embed(unique)
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            sim = sbert_service.cosine_similarity(vectors[i], vectors[j])
            if sim >= CONFUSE_THRESHOLD:
                pairs.append((unique[i], unique[j], round(sim, 3)))
    pairs.sort(key=lambda item: item[2], reverse=True)
    logger.info("易混淆语义推理完成：%d 个概念，发现 %d 对易混淆候选", len(unique), len(pairs))
    return pairs


def analogy(concept_a: str, concept_b: str, concept_c: str) -> Optional[str]:
    """执行向量类比 A - B + C，返回最相近的已知概念。

    Args:
        concept_a: 类比词 A。
        concept_b: 类比词 B。
        concept_c: 类比词 C（从概念库中取）。

    Returns:
        类比得出的目标概念；嵌入失败时返回 None。
    """
    try:
        vec_a = sbert_service.embed([concept_a])[0]
        vec_b = sbert_service.embed([concept_b])[0]
        vec_c = sbert_service.embed([concept_c])[0]
        target = vec_a - vec_b + vec_c
        return str(np.argmax(target))
    except RuntimeError:
        return None


def build_differentiation_hint(pair_a: str, pair_b: str, similarity: float) -> str:
    """为一对易混淆概念生成辨析提示文本。

    Args:
        pair_a: 概念 A。
        pair_b: 概念 B。
        similarity: 二者余弦相似度。

    Returns:
        面向用户的辨析提示语。
    """
    return (
        f"「{pair_a}」与「{pair_b}」语义高度相近（相似度 {similarity:.2f}），"
        "作答时注意区分二者的适用前提与边界条件。"
    )


def build_semantic_space(concepts: List[str]) -> Dict[str, List[float]]:
    """将概念列表映射为关键词到向量的语义空间字典。

    Args:
        concepts: 概念列表。

    Returns:
        {概念: 384 维向量列表} 的字典。
    """
    unique = list(dict.fromkeys(concepts))
    vectors = sbert_service.embed(unique)
    return {name: vec.tolist() for name, vec in zip(unique, vectors)}