"""F2-1 / F2-2：纯规则的线索词悬停统计与反常识词对挖掘。

F2-1：统计关键词在题目文本中的出现频次，输出认知摩擦指数（出现越多越可能成为
认知摩擦点）。无真实悬停数据时，以题干词频近似。
F2-2：从题目文本中挖掘语义对立词对（如 光滑 vs 粗糙、恒力 vs 变力）。
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# 常见反义/对立词对（可扩展）
_ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("光滑", "粗糙"),
    ("恒力", "变力"),
    ("静止", "运动"),
    ("恒定", "变化"),
    ("加速", "减速"),
    ("串联", "并联"),
    ("增大", "减小"),
    ("上升", "下降"),
    ("正电荷", "负电荷"),
    ("电场", "磁场"),
    ("实像", "虚像"),
    ("自由落体", "抛体运动"),
]


def hesitation_weights(question_text: str) -> Dict[str, float]:
    """统计题干中线索词的悬停权重（F2-1）。

    以关键词出现频次归一化，输出认知摩擦指数 0~1。

    Args:
        question_text: 题干全文。

    Returns:
        {词组: 权重} 字典，权重越高越可能是认知摩擦点。
    """
    from collections import Counter

    # 以题干中出现的双字词与反义词对作为候选线索词
    tokens = re.findall(r"[\u4e00-\u9fa5]{2,}", question_text)
    counter = Counter(tokens)
    total = sum(counter.values()) or 1
    weights = {word: round(count / total, 3) for word, count in counter.most_common(8)}
    return weights


def antonym_pairs(question_text: str) -> List[Tuple[str, str]]:
    """从题干文本挖掘语义对立词对（F2-2）。

    Args:
        question_text: 题干全文。

    Returns:
        题干中出现的反义词对列表。
    """
    found: List[Tuple[str, str]] = []
    for pair_a, pair_b in _ANTONYM_PAIRS:
        if pair_a in question_text or pair_b in question_text:
            found.append((pair_a, pair_b))
    return found