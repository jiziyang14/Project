# -*- coding: utf-8 -*-
"""划词预测推理：给一道题 → 生成候选 → 用 huaci_rerank 打分 → 建议该划的词。

模型不存在或关闭时优雅回退到候选生成的启发式 rank，行为同旧逻辑。
特征口径与训练完全一致（huaci_candidates.huaci_to_features）。
"""
from __future__ import annotations

import pickle
from typing import List, Optional

from core.config import root_path
from modules.semantic import huaci_candidates as hc

_RERANK = {}  # subject -> model payload（分学科各训一个模型）
_RERANK_ENABLED = True


def _load(subject: str = "数学"):
    global _RERANK
    if subject in _RERANK or not _RERANK_ENABLED:
        return _RERANK.get(subject)
    path = root_path(f"data/models/huaci_{subject}.pkl")
    if path.exists():
        try:
            with open(path, "rb") as fh:
                _RERANK[subject] = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            print("划词模型(%s)加载失败，回退规则评分：%s" % (subject, exc))
            _RERANK[subject] = None
    else:
        _RERANK[subject] = None
    return _RERANK.get(subject)


def invalidate() -> None:
    """训练后清除缓存，让推理立即用新模型。"""
    global _RERANK
    _RERANK = {}


def score(cand: dict, subject: str = "数学") -> float:
    """用对应学科模型对单候选打分；模型不可用时回退启发式 rank。"""
    m = _load(subject)
    if m is None:
        return float(cand.get("rank", 12))
    proba = m["model"].predict_proba([hc.huaci_to_features(cand)])[0]
    # 应划度 = 2·P(2) + 1·P(1)；P(2)与P(1)越高的越建议划
    return float(2.0 * proba[2] + 1.0 * proba[1])


def predict_question(question: str, options: Optional[List[str]] = None,
                     subject: str = "数学", top_k: int = 12) -> List[dict]:
    """给一道题返回"建议划取的候选"，按应划度降序，附模型预测 label 与概率。"""
    cands = hc.generate_question_candidates(question, options or [], subject=subject, top_k=20)
    m = _load(subject)
    for c in cands:
        c["score"] = round(score(c, subject), 4)
        if m is not None:
            proba = m["model"].predict_proba([hc.huaci_to_features(c)])[0]
            c["label_pred"] = int(proba.argmax())
            c["proba"] = [round(float(x), 3) for x in proba]
    desc = sorted(cands, key=lambda c: c["score"], reverse=True)
    for rank, c in enumerate(desc[:top_k], start=1):
        c["rank"] = rank
    return desc[:top_k]


def status() -> dict:
    out = {}
    for subj in ("数学", "语文"):
        m = _load(subj)
        out[subj] = {
            "ready": m is not None,
            "version": m.get("version") if m else None,
            "metrics": m.get("metrics") if m else None,
        }
    return out