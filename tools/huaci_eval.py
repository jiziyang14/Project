# -*- coding: utf-8 -*-
"""划词模型效果评估：在同一模型（无学科特征、两科共用）下，
分别看数学/语文各自的 holdout 表现，并给出混淆矩阵与特征重要性。

用法：py -3.12 -m tools.huaci_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import root_path  # noqa: E402
from modules.semantic import huaci_candidates as hc  # noqa: E402

SUBJECTS = ["数学", "语文"]
LABELS = {0: "label0", 1: "label1", 2: "label2"}


def _load() -> list:
    rows = []
    for subj in SUBJECTS:
        p = Path(root_path(f"data/huaci/{subj}/train_samples.jsonl"))
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") not in (0, 1, 2):
                continue
            r["_subj"] = subj
            rows.append(r)
    return rows


def _acc_f1(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return accuracy_score(y, p), f1_score(y, p, average="weighted")


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    rows = _load()
    if len(rows) < 20:
        print("标注样本不足")
        return 1

    X = np.array([hc.huaci_to_features(r) for r in rows], dtype=np.float64)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    subj = np.array([r["_subj"] for r in rows], dtype=object)
    fn = hc.huaci_feature_names()

    clf = GradientBoostingClassifier(
        n_estimators=180, max_depth=3, min_samples_leaf=8, learning_rate=0.08,
        subsample=0.85, random_state=42,
    )
    clf.fit(X, y)  # 与线上一致：全量训练后评估 → 反映"部署后真实可及"的分布指标

    pred = clf.predict(X)
    print("== 全量分布（自评，非 holdout，仅看吻合度） ==")
    imp = sorted(zip(fn, clf.feature_importances_), key=lambda x: -x[1])
    print("特征重要性：", "、".join(f"{f}={i:.3f}" for f, i in imp[:6]))
    print("总体 ACC %.4f · F1(weighted) %.4f" % (np.mean(pred == y), f1_score(y, pred, average="weighted")))
    print("混淆矩阵（行=真实 0/1/2，列=预测）：")
    print(confusion_matrix(y, pred))

    print("\n== 分科 holdout（每科内 80/20，分层，10 次平均） ==")
    per_label = {s: {0: 0, 1: 0, 2: 0} for s in SUBJECTS}
    for _ in range(10):
        for s in SUBJECTS:
            idx = np.where(subj == s)[0]
            if len(idx) < 30:
                continue
            Xs, ys, ss = X[idx], y[idx], subj[idx]
            Xtr, Xva, ytr, yva = train_test_split(Xs, ys, test_size=0.2, random_state=np.random.RandomState(seed=42).randint(0, 9999), stratify=ys)
            m = GradientBoostingClassifier(
                n_estimators=180, max_depth=3, min_samples_leaf=8,
                learning_rate=0.08, subsample=0.85, random_state=42)
            m.fit(Xtr, ytr)
            pv = m.predict(Xva)
            for lbl in (0, 1, 2):
                per_label[s][lbl] += float(f1_score(yva == lbl, pv == lbl))
    print("%-5s %8s %8s %8s" % ("学科", "label0 F1", "label1 F1", "label2 F1"))
    for s in SUBJECTS:
        n = sum(1 for r in rows if r["_subj"] == s)
        print("%-5s %10.3f %10.3f %10.3f  (样本 %d)"
              % (s, per_label[s][0] / 10, per_label[s][1] / 10, per_label[s][2] / 10, n))

    # 分科类别分布（便于理解）
    print("\n== 每科标注分布 ==")
    for s in SUBJECTS:
        m = sum(1 for r in rows if r["_subj"] == s)
        d = {l: sum(1 for r in rows if r["_subj"] == s and r["label"] == l) for l in (0, 1, 2)}
        print("%-5s n=%d  %s" % (s, m, {k: "%d(%.0f%%)" % (v, v / m * 100) for k, v in d.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())