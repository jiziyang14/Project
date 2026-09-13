# -*- coding: utf-8 -*-
"""划词预测模型训练：读 data/huaci/{学科}/train_samples.jsonl，训练 0/1/2 分类器。

输出：
  data/models/huaci_rerank.pkl  —— dict{model, features, version, trained_at, metrics}
  data/models/huaci_rerank.json  —— 同内容元数据（供前端展示）

特征与推理完全一致（modules.semantic.huaci_candidates.huaci_to_features），
避免训练/推理口径漂移。label 仅作训练目标，绝不进入特征。

用法：py -3.12 -m tools.huaci_train
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import root_path  # noqa: E402
from modules.semantic import huaci_candidates as hc  # noqa: E402

SUBJECTS = ["数学", "语文"]
LABELS = {0: "label0", 1: "label1", 2: "label2"}


def _pkl(subj: str) -> Path:
    return Path(root_path(f"data/models/huaci_{subj}.pkl"))


def _load(subjects) -> list:
    """读多个学科的划词训练样本。"""
    rows = []
    for subj in subjects:
        p = Path(root_path(f"data/huaci/{subj}/train_samples.jsonl"))
        if not p.exists():
            print(f"[跳过] 无训练样本：{p}")
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") not in (0, 1, 2):
                continue
            rows.append(r)
    return rows


def train(subjects=None) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    subjects = subjects or SUBJECTS
    results = {}
    for subj in subjects:
        rows = _load([subj])
        if len(rows) < 20:
            print(f"[跳过] {subj}：标注样本不足 {len(rows)}")
            continue

        X = np.array([hc.huaci_to_features(r) for r in rows], dtype=np.float64)
        y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
        fn = hc.huaci_feature_names()

        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        clf = GradientBoostingClassifier(
            n_estimators=180, max_depth=3, min_samples_leaf=8, learning_rate=0.08,
            subsample=0.85, random_state=42,
        )
        clf.fit(Xtr, ytr)

        acc = float(accuracy_score(yva, clf.predict(Xva)))
        f1_w = float(f1_score(yva, clf.predict(Xva), average="weighted"))
        f1_m = float(f1_score(yva, clf.predict(Xva), average="macro"))
        report = classification_report(yva, clf.predict(Xva), labels=[0, 1, 2],
                                       target_names=list(LABELS.values()), output_dict=True, zero_division=0)
        imp = sorted(zip(fn, clf.feature_importances_), key=lambda x: -x[1])

        # 全量再训一版用于线上
        clf.fit(X, y)

        metrics = {
            "subject": subj,
            "n_samples": len(rows),
            "holdout_acc": round(acc, 4),
            "holdout_f1_weighted": round(f1_w, 4),
            "holdout_f1_macro": round(f1_m, 4),
            "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items() if kk in ("precision", "recall", "f1-score", "support")}
                          for k, v in report.items() if k in LABELS.values()},
            "top_features": [[str(f), round(float(i), 4)] for f, i in imp[:5]],
        }

        payload = {
            "subject": subj,
            "model": clf,
            "features": fn,
            "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics,
        }
        out = _pkl(subj)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(payload, fh)
        out.with_suffix(".json").write_text(
            json.dumps({k: payload[k] for k in ("subject", "version", "features", "trained_at", "metrics")},
                       ensure_ascii=False, indent=2), encoding="utf-8")

        print("划词模型已训练 -> %s" % out)
        print("  %s · 样本 %d · holdout ACC %.4f · F1(weighted) %.4f" % (subj, len(rows), acc, f1_w))
        print("  最重要特征：", "、".join(f"{f}={i:.3f}" for f, i in imp[:5]))
        for k, v in metrics["per_class"].items():
            print("    %-7s P=%.3f R=%.3f F1=%.3f n=%d" % (k, v["precision"], v["recall"], v["f1-score"], v["support"]))
        results[subj] = metrics
    return results


def main() -> int:
    try:
        res = train()
        print("\n全部学科模型完成：", "、".join(res) or "无")
    except Exception as exc:
        print("训练失败：%s" % exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())