"""F2-6 题型语义聚类：SBERT 嵌入 + K-Means + 轮廓系数定 K。

将全量题目题干按知识主题/题型自动分组，簇命名由 TF-IDF 关键词拼接生成。
触发条件：题库 ≥ 5 道题；每次新增题目时增量更新。
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from core.schemas import ClusterReport, Question
from modules.semantic import sbert_service

logger = logging.getLogger(__name__)

MIN_QUESTIONS = 5
K_RANGE = range(3, 9)  # 轮廓系数自动确定的范围 3~8


def cluster_questions(questions: List[Question]) -> List[ClusterReport]:
    """对题目列表执行题型聚类。

    Args:
        questions: 全量题目列表，需 ≥5 道。

    Returns:
        聚类报告列表，按占比降序。

    Raises:
        ValueError: 题目数不足触发阈值。
    """
    if len(questions) < MIN_QUESTIONS:
        raise ValueError(f"题目数不足 {MIN_QUESTIONS} 道，暂无法聚类")
    texts = [q.question_text for q in questions]
    vectors = sbert_service.embed(texts)

    best_k = _select_k(vectors)
    labels, centers = _kmeans_fit(vectors, best_k)

    reports: List[ClusterReport] = []
    for cluster_id in range(best_k):
        member_ids = [q.question_id for q, label in zip(questions, labels) if label == cluster_id]
        if not member_ids:
            continue
        keywords = _tfidf_keywords(texts, labels, cluster_id)
        reports.append(
            ClusterReport(
                cluster_id=cluster_id,
                cluster_label="、".join(keywords[:3]) or f"类型{cluster_id}",
                representative_questions=[member_ids[0]],
                keywords=keywords,
                percentage=len(member_ids) / len(questions),
                question_ids=member_ids,
            )
        )
    reports.sort(key=lambda r: r.percentage, reverse=True)
    logger.info("题型聚类完成：%d 道题分为 %d 类", len(questions), len(reports))
    return reports


def _select_k(vectors: np.ndarray) -> int:
    """用轮廓系数在 3~8 范围内自动选定 K。

    Args:
        vectors: 题目嵌入矩阵。

    Returns:
        轮廓系数最大对应的 K 值；K 范围内无法聚类时取 3。
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_score = 3, -1.0
    for k in K_RANGE:
        if k >= len(vectors):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _kmeans_fit(vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """执行 K-Means 并返回标签与簇中心。

    Args:
        vectors: 题目嵌入矩阵。
        k: 簇数量。

    Returns:
        (标签数组, 簇中心矩阵)。
    """
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(vectors)
    return labels, model.cluster_centers_


def _tfidf_keywords(texts: List[str], labels: np.ndarray, cluster_id: int) -> List[str]:
    """对指定簇用 TF-IDF 提取 Top 关键词。

    Args:
        texts: 全部题干。
        labels: K-Means 标签。
        cluster_id: 目标簇编号。

    Returns:
        按权重降序的关键词列表。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=16)
        matrix = vectorizer.fit_transform(texts)
        cluster_mask = labels == cluster_id
        cluster_center = np.asarray(matrix[cluster_mask].mean(axis=0)).ravel()
        order = np.argsort(cluster_center)[::-1]
        names = vectorizer.get_feature_names_out()
        return [names[i] for i in order[:3] if cluster_center[i] > 0]
    except ValueError:
        return []