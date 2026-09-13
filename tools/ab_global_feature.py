# -*- coding: utf-8 -*-
"""A/B 对比：文章关键词重排模型「加全局词频特征 vs 不加」。

结论判定唯一依据 = 同一批独立 27 篇文章上的 Hit@1 / P@8 / Gold 召回。
公平性保证：
- 训练样本完全一样（data/training/labels.jsonl）；
- 80/20 切分完全一样（random_state=42），且两模型共用同一份 train/val；
- 模型结构与超参完全一样（GradientBoostingRegressor 同参数、同随机种子）；
- 两模型在评估阶段共用线上完全相同的打分/融合/去重逻辑，仅特征维度不同（11 vs 13）。

产出：
- 控制台打印对比表 + 判定
- Report/全局特征A-B实验报告.md 落一份可读报告（不烂在文件夹）

用法：py -3.12 -m tools.ab_global_feature
"""
from __future__ import annotations

import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import root_path  # noqa: E402
from modules.semantic import keywords as kw  # noqa: E402
from tools import train_rerank  # noqa: E402  # 复用 _load_labeled

LBL = root_path("data/training/labels.jsonl")
TEST_SET = root_path("data/training/kw_testset.json")
GWF = root_path("data/models/global_wordfish.pkl")
TOP_N = 8
REPORT = root_path("Report/全局特征A-B实验报告.md")

# 全局特征名（追加在原有 11 维之后）
NAME_NEW = ["global_idf", "global_candfreq"]

_global_table = None
# 捕获原始特征函数，避免 evaluation 打补丁后 _to_features_ab 内部递归
_base_to_features = kw._to_features


def _load_global():
    global _global_table
    if _global_table is None and GWF.exists():
        with open(GWF, "rb") as fh:
            _global_table = pickle.load(fh)
    return _global_table


def _global_features(word: str):
    """返回该词的两个全局特征：global_idf=稀有度(log尺)，global_candfreq=全库累计频次(log1p)。"""
    g = _load_global()
    if g is None:
        return [0.0, 0.0]
    n = max(1, g["n_docs"])
    df = g["doc_freq"].get(word, 0)
    tf = g["total_freq"].get(word, 0)
    idf = math.log(n / (df + 1.0))
    return [idf, math.log1p(tf)]


def _to_features_ab(item: dict) -> list:
    base = _base_to_features(item)
    return base + _global_features(str(item.get("word") or item.get("kw", "")))


def _feature_names_ab() -> list:
    return kw._feature_names() + NAME_NEW


def _load_samples():
    rows = train_rerank._load_labeled(str(LBL))
    return rows


def train_model(X, y):
    from sklearn.ensemble import GradientBoostingRegressor
    Xtr, Xva, ytr, yva = X, X[:int(len(X) * 0.8)], y[:int(len(y) * 0.8)], y[int(len(X) * 0.8):]
    # 与 train_rerank 相同的随机切分
    from sklearn.model_selection import train_test_split
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GradientBoostingRegressor(
        n_estimators=140, max_depth=3, min_samples_leaf=5, learning_rate=0.08,
        random_state=42, loss="squared_error",
    )
    clf.fit(Xtr, ytr)
    from scipy.stats import spearmanr
    pred = clf.predict(Xva)
    corr, _ = spearmanr(yva, pred)
    acc = float(np.mean(np.abs(np.round(pred * 2) - yva * 2) < 0.5))
    # 全量再训一版用于评估
    clf.fit(X, y)
    return clf, {"spearman": round(float(corr), 4), "class_acc": round(acc, 4)}


def eval_on_test(model, use_global: bool) -> list:
    """复用线上 loading/打分/去重流程，仅替换模型与特征维度。"""
    test_rows = json.loads(TEST_SET.read_text(encoding="utf-8"))
    if use_global:
        kw._to_features = _to_features_ab
        names = _feature_names_ab()
    else:
        kw._to_features = _base_to_features
        names = kw._feature_names()
    kw._RERANK = {"model": model, "features": names, "version": "ab_" + ("global" if use_global else "base")}
    kw._RERANK_ENABLED = True
    out = []
    for tr in test_rows:
        title = tr["title"].split("_", 1)[-1]
        res = kw.extract_keywords_full(tr["text"], top_n=TOP_N, title=title)
        tops = [k["word"] for k in res["keywords"]]
        gold = set(tr["gold"]); rel = set(tr["related"])
        hit_gold = [w for w in tops if w in gold]
        out.append({
            "title": tr["title"], "top": tops, "gold": tr["gold"], "hit_gold": hit_gold,
            "n_gold": len(gold),
            "p8_gold": len(hit_gold) / TOP_N,
            "p8_rel": sum(1 for w in tops if w in gold or w in rel) / TOP_N,
            "hit1": bool(tops) and tops[0] in gold,
        })
    return out


def summ(rows):
    n = max(1, len(rows))
    return {
        "hit1": sum(1 for r in rows if r["hit1"]) / n,
        "any_hit": sum(1 for r in rows if r["hit_gold"]) / n,
        "p8_gold": sum(r["p8_gold"] for r in rows) / n,
        "p8_rel": sum(r["p8_rel"] for r in rows) / n,
        "recall": sum(len(r["hit_gold"]) for r in rows) / max(1, sum(r["n_gold"] for r in rows)),
    }


def main() -> int:
    samples = _load_samples()
    if len(samples) < 20:
        print("标注样本不足，无法训练。")
        return 1

    y = np.array([int(c["label"]) / 2.0 for _, c in samples], dtype=np.float64)
    X_base = np.array([kw._to_features(c) for _, c in samples], dtype=np.float64)
    X_ab = np.array([_to_features_ab(c) for _, c in samples], dtype=np.float64)

    print("训练样本：%d（文档 %d） 特征：基线=%d / +全局=%d" % (len(samples), len({d for d, _ in samples}), X_base.shape[1], X_ab.shape[1]))

    model_base, val_base = train_model(X_base, y)
    model_ab, val_ab = train_model(X_ab, y)

    saved_rerank, saved_enabled = kw._RERANK, kw._RERANK_ENABLED
    try:
        rows_base = eval_on_test(model_base, use_global=False)
        rows_ab = eval_on_test(model_ab, use_global=True)
    finally:
        kw._to_features = _base_to_features
        kw._RERANK, kw._RERANK_ENABLED = saved_rerank, saved_enabled

    m_base, m_ab = summ(rows_base), summ(rows_ab)
    n_t = len(rows_base)

    def line(label, m, val):
        return ("%s | 命中=%.0f%% | 篇命中=%.0f%% | P@8(关键)=%.2f | P@8(相关)=%.2f | Gold召回=%.2f | val.Sp=%.3f | val.Acc=%.0f%%"
                % (label, m["hit1"] * 100, m["any_hit"] * 100, m["p8_gold"], m["p8_rel"], m["recall"], val["spearman"], val["class_acc"] * 100))

    header = "测试集=%d 篇（独立，不参与训练） Top%d" % (n_t, TOP_N)
    # 判定：采用 +全局再重训，还是维持基线
    improves = {}
    for k in ("hit1", "any_hit", "p8_gold", "p8_rel", "recall"):
        improves[k] = m_ab[k] - m_base[k]
    win = sum(1 for k in improves if improves[k] > 0)
    lose = sum(1 for k in improves if improves[k] < 0)
    verdict = "建议采用 +全局特征 并重训（%d/5 项提升）" % win if win > lose else ("建议保持基线（全局特征未带来明显提升，%d胜%d负）" % (win, lose))

    print("=" * 78)
    print("A/B 实验：文章关键词重排（加全局词频特征 vs 不加）")
    print(header)
    print("-" * 78)
    print(line("基线(11维)", m_base, val_base))
    print(line("+全局(13维)", m_ab, val_ab))
    print("-" * 78)
    for k in ("hit1", "any_hit", "p8_gold", "p8_rel", "recall"):
        d = improves[k] * (100 if k in ("hit1", "any_hit") else 1)
        print("  Δ%-8s %+.2f" % (k, d))
    print("判定：%s" % verdict)

    # 落一份可读报告，避免"烂在文件夹里没人管"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    rows_md = ["| 指标 | 基线(11维) | +全局(13维) | 差Δ |",
               "|---|---|---|---|"]
    for k, name in (("hit1", "Hit@1"), ("any_hit", "篇命中"), ("p8_gold", "P@8(关键)"), ("p8_rel", "P@8(相关)"), ("recall", "Gold召回")):
        a_, b_ = m_base[k], m_ab[k]
        d_ = b_ - a_
        unit = 1.0 if k in ("hit1", "any_hit") else 1.0
        rows_md.append("| %s | %s | %s | %s |" % (
            name,
            ("%.0f%%" % (a_ * 100)) if k in ("hit1", "any_hit") else "%.2f" % a_,
            ("%.0f%%" % (b_ * 100)) if k in ("hit1", "any_hit") else "%.2f" % b_,
            ("%+.0fpp" % (d_ * 100)) if k in ("hit1", "any_hit") else "%+.2f" % d_))
    top_change = ""
    for a_, b_ in zip(rows_base, rows_ab):
        if a_["hit1"] != b_["hit1"]:
            top_change += "\n- `%s`：基线 Top1=%s → 加全局 Top1=%s（Gold=%s）" % (a_["title"], a_["top"][0] if a_["top"] else "—", b_["top"][0] if b_["top"] else "—", "/".join(a_["gold"]))
    REPORT.write_text(
        "# 全局词频特征 A/B 实验报告\n\n"
        "> 一键重现：`py -3.12 -m tools.ab_global_feature`　·　%s\n\n"
        "## 结论\n**%s**\n\n"
        "## 对比表\n\n%s\n\n"
        "## 实验设置（公平性）\n"
        "- 训练样本：`data/training/labels.jsonl`（完全相同）；80/20 切分 random_state=42（两模型共用同一份 train/val）。\n"
        "- 模型：GradientBoostingRegressor 同参数同种子，仅特征维度不同（11 vs 11+2）。\n"
        "- 评估：独立 27 篇测试集（不参与训练），复用线上打分/融合/去重，仅换模型。\n"
        "- 新增特征：`global_idf`=稀有度 log(N/(doc_freq+1))，`global_candfreq`=全库累计频次 log1p(total_freq)；词频表来自 `data/all_docs.txt` 全 108281 篇扫描。\n\n"
        "## 验证指标（模型自评 80/20 留出）\n"
        "- 基线：val.Spearman=%.3f / 档位命中=%.0f%%\n"
        "- 加全局：val.Spearman=%.3f / 档位命中=%.0f%%\n\n"
        "## Top1 变化明细（基线 vs 加全局 不一致的样本）\n%s\n"
        % (header.replace("|", "\|"), verdict,
           "\n".join(rows_md),
           val_base["spearman"], val_base["class_acc"] * 100,
           val_ab["spearman"], val_ab["class_acc"] * 100,
           top_change or "\n（无差异）"), encoding="utf-8"
    )
    print("报告已写入：%s" % REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())