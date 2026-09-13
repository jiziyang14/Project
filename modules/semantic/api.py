"""语义层对外隔离接口。

其他模块只能通过本 api 调用 F2-3/F2-5/F2-6，不直接接触内部实现。
"""
from __future__ import annotations

from typing import List, Tuple

from core.schemas import ClusterReport, ErrorClusterReport, Question
from modules.semantic import confusion, error_cluster, friction, keywords, question_cluster, sbert_service


def embed_texts(texts: List[str]):
    """对文本列表计算语义向量（F2-3/F2-4/F2-6 共用的底层服务）。

    Args:
        texts: 待嵌入文本列表。

    Returns:
        shape (N, 384) 的归一化嵌入矩阵。
    """
    return sbert_service.embed(texts)


def confusable_pairs(concepts: List[str]) -> List[Tuple[str, str, float]]:
    """挖掘易混淆概念对（F2-3）。

    Args:
        concepts: 知识点/中心词列表。

    Returns:
        按相似度降序的 (A, B, 相似度) 列表。
    """
    return confusion.find_confusable_pairs(concepts)


def extract_article_keywords(text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    """提取文章中心词（F2-5）。

    Args:
        text: 文章文本。
        top_n: 返回数量。

    Returns:
        (关键词, 权重) 列表。
    """
    return keywords.extract_keywords(text, top_n=top_n)


def cluster_question_types(questions: List[Question]) -> List[ClusterReport]:
    """执行题型聚类（F2-6）。

    Args:
        questions: 全量题目。

    Returns:
        聚类报告列表。

    Raises:
        ValueError: 题目数不足 5。
    """
    return question_cluster.cluster_questions(questions)


def cluster_wrong_answers(questions: List[Question]) -> List[ErrorClusterReport]:
    """执行错因聚类（F2-4）。

    Args:
        questions: 错题列表。

    Returns:
        错因报告列表。

    Raises:
        ValueError: 错题数不足 5 或缺少错误答案。
    """
    return error_cluster.cluster_wrong_answers(questions)


def hesitation_weights(question_text: str) -> dict:
    """线索词悬停权重（F2-1）。

    Args:
        question_text: 题干文本。

    Returns:
        {词组: 权重} 字典。
    """
    return friction.hesitation_weights(question_text)


def antonym_pairs(question_text: str) -> List[Tuple[str, str]]:
    """反常识词对挖掘（F2-2）。

    Args:
        question_text: 题干文本。

    Returns:
        题干中出现的反义词对。
    """
    return friction.antonym_pairs(question_text)