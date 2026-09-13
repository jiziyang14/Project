"""F2-4 错误原因聚类：SBERT 嵌入 + DBSCAN 聚类。

将错题的错误答案文本聚类为错因类型，输出标签：逻辑断裂型 / 前提误解型 / 创新型。
触发条件：错题库 ≥ 5 道且含错误答案文本。
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from core.schemas import ErrorClusterReport, Question
from modules.semantic import sbert_service

logger = logging.getLogger(__name__)

MIN_WRONG = 5
LABEL_POOL = ["逻辑断裂型", "前提误解型", "创新型"]

# DBSCAN 参数
EPS = 0.5
MIN_SAMPLES = 3


def cluster_wrong_answers(questions: List[Question]) -> List[ErrorClusterReport]:
    """对错题的错误答案执行错因聚类。

    Args:
        questions: 错题列表（需 ≥5 道且含错误答案文本）。

    Returns:
        错因报告列表，按占比降序。

    Raises:
        ValueError: 触发条件不满足。
    """
    wrong_with_answer = [q for q in questions if q.user_answer and q.user_answer.strip()]
    if len(wrong_with_answer) < MIN_WRONG:
        raise ValueError(f"带错误答案的错题不足 {MIN_WRONG} 道，暂无法聚类")

    answers = [q.user_answer for q in wrong_with_answer]
    vectors = sbert_service.embed(answers)

    labels = _dbscan_fit(vectors)
    # 噪声点（-1）单独归为"其他"
    reports = _build_reports(labels, wrong_with_answer)
    reports.sort(key=lambda r: r.percentage, reverse=True)
    logger.info("错因聚类完成：%d 道错题，%d 个错因簇", len(wrong_with_answer), len(reports))
    return reports


def _dbscan_fit(vectors: np.ndarray) -> np.ndarray:
    """执行 DBSCAN 聚类。

    Args:
        vectors: 错误答案嵌入矩阵。

    Returns:
        每个样本的簇标签，-1 表示噪声。
    """
    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="cosine")
    return model.fit_predict(vectors)


def _build_reports(labels: np.ndarray, questions: List[Question]) -> List[ErrorClusterReport]:
    """将簇标签整理为报告列表，噪声点归入"其他"。

    Args:
        labels: DBSCAN 标签。
        questions: 与标签一一对应的错题。

    Returns:
        错因报告列表。
    """
    unique_labels = sorted(set(int(label) for label in labels if label >= 0))
    reports: List[ErrorClusterReport] = []
    for cluster_id in unique_labels:
        member_ids = [q.question_id for q, lb in zip(questions, labels) if int(lb) == cluster_id]
        # 标签按池循环取用，保证有语义化名称
        name = LABEL_POOL[cluster_id % len(LABEL_POOL)]
        reports.append(
            ErrorClusterReport(
                cluster_id=cluster_id,
                cluster_label=name,
                percentage=len(member_ids) / len(questions),
                wrong_answer_ids=member_ids,
            )
        )
    noise_ids = [q.question_id for q, lb in zip(questions, labels) if int(lb) < 0]
    if noise_ids:
        reports.append(
            ErrorClusterReport(
                cluster_id=-1,
                cluster_label="其他",
                percentage=len(noise_ids) / len(questions),
                wrong_answer_ids=noise_ids,
            )
        )
    return reports