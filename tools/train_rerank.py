"""训练文章关键词有监督重排器。

用法：
    python -m tools.train_rerank            # 读默认标注底稿，产出 data/models/kw_rerank.pkl
    python -m tools.train_rerank --labels data/training/labels.jsonl --out data/models/kw_rerank.pkl

输入：标注底稿 JSONL（backend/training.save_labels 产出，候选含 label 0/1/2）。
输出：pickle 文件（dict: model / features / version / trained_at / metrics 等），
      keywords.swift 推理时存在即自动启用该模型替换手写加权融合；
      同时写 kw_rerank.json 供前端展示。

特征与 modules/semantic/keywords._to_features 完全一致，保证训练/推理口径统一。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from core.config import root_path
from modules.semantic import keywords as kw


def _load_labeled(labels_path: str):
    rows = []
    p = Path(labels_path)
    if not p.exists():
        print(f"[错误] 找不到标注底稿：{p}")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # 从存储的原文+标题计算上下文特征（in_title / first_pos_norm / sent_cov），
            # 与推理时 extract_keywords_full 的口径保持一致。
            text = row.get("text", "")
            title = row.get("title", "")
            tokens, _ = kw._segment(text)
            sentences = [s for s in re.split(r"[。！？!?；;]", text) if s.strip()]
            title_words: set = set()
            if title and title.strip():
                title_tokens, _ = kw._segment(title)
                title_words = {w for w in title_tokens if 2 <= len(w) <= 6}
            for c in row.get("candidates", []):
                if c.get("label") is not None and c.get("label") in (0, 1, 2):
                    c.update(kw._context_features(c["kw"], tokens, sentences, title_words))
                    rows.append((row["doc_id"], c))
    return rows


def train(labels_path: str, out_path: str, max_samples: int = 0) -> dict:
    """核心训练入口：读标注底稿 → 训练 → 落盘模型与元数据，返回指标。"""
    from modules.semantic import keywords as kw  # 复用统一特征口径

    samples = _load_labeled(labels_path)
    if max_samples:
        samples = samples[:max_samples]
    if len(samples) < 20:
        print(f"[提示] 有效标注样本偏少（{len(samples)}），建议到 200+ 以获得稳定效果")

    X = np.array([kw._to_features(c) for _, c in samples], dtype=np.float64)
    y = np.array([int(c["label"]) / 2.0 for _, c in samples], dtype=np.float64)  # [0,0.5,1]

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from scipy.stats import spearmanr

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = GradientBoostingRegressor(
        n_estimators=140, max_depth=3, min_samples_leaf=5, learning_rate=0.08,
        random_state=42, loss="squared_error",
    )
    clf.fit(Xtr, ytr)

    pred_va = clf.predict(Xva)
    rmse = float(np.sqrt(mean_squared_error(yva, pred_va)))
    corr, _ = spearmanr(yva, pred_va)
    acc = float(np.mean(np.abs(np.round(pred_va * 2) - yva * 2) < 0.5))

    # 完整数据再训练一版（用全量获得更好的线上表现）
    clf.fit(X, y)

    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics = {
        "n_docs": len({d for d, _ in samples}),
        "n_samples": len(samples),
        "val_rmse": round(rmse, 4),
        "val_spearman": round(corr, 4),
        "val_class_acc": round(acc, 4),
    }
    payload = {
        "model": clf,
        "features": kw._feature_names(),
        "version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_docs": metrics["n_docs"],
        "n_samples": metrics["n_samples"],
        "metrics": metrics,
        "blend": "gradient_boost_reranker",
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        import pickle

        pickle.dump(payload, fh)

    meta = {k: payload[k] for k in ("version", "features", "trained_at", "n_docs", "n_samples", "metrics", "blend")}
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 使已加载的重排模型缓存失效，让正在运行的服务立即改用新模型
    try:
        kw.invalidate_rerank()
    except Exception as exc:  # 即使失效失败也不阻塞训练返回
        print("[提示] 清除模型缓存失败：%s" % exc)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="训练文章关键词重排器")
    ap.add_argument("--labels", default=str(root_path("data/training/labels.jsonl")))
    ap.add_argument("--out", default=str(root_path("data/models/kw_rerank.pkl")))
    ap.add_argument("--max-samples", type=int, default=0, help="最多使用标注样本数（调试用）")
    args = ap.parse_args()

    metrics = train(args.labels, args.out, args.max_samples)

    from modules.semantic import keywords as kw

    with open(args.out, "rb") as fh:
        import pickle

        payload = pickle.load(fh)
    clf = payload["model"]
    print("=" * 60)
    print("关键词重排模型训练完成")
    print(f"  标注样本 : {metrics['n_samples']}（文档 {metrics['n_docs']} 篇）")
    print(f"  验证 RMSE: {metrics['val_rmse']:.4f}")
    print(f"  验证 Spearman: {metrics['val_spearman']:.4f}")
    print(f"  档位命中率: {metrics['val_class_acc']:.1%}")
    print("  特征重要性 (Top5):")
    fi = sorted(zip(kw._feature_names(), clf.feature_importances_), key=lambda t: -t[1])[:5]
    for name, imp in fi:
        print(f"    - {name}: {imp:.3f}")
    print(f"  模型已保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())