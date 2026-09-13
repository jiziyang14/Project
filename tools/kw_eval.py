# -*- coding: utf-8 -*-
"""关键词提取评估入口（可复用模块）。

用法：
    python -m tools.kw_eval                 # 打印汇总
    python -m tools.kw_eval --json          # 输出 JSON（供后端/前端使用）

设计要点（合规评估）：
- 测试集固定为 data/training/kw_testset.json：27 篇全新真实文章（data/testset/），
  从未出现在 articles.jsonl / labels.jsonl，天然不参与训练，避免测试集重叠高估。
- 评估直接使用「正式部署模型」data/models/kw_rerank.pkl（未重训临时模型），
  反映线上真实提取效果；并与无模型规则版基线对比。
- 评估过程不修改正式 kw_rerank.pkl / labels.jsonl / articles.jsonl；
  结束后恢复 keywords 模块的重排模型缓存，不影响线上服务。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import root_path  # noqa: E402
from modules.semantic import keywords as kw  # noqa: E402

LBL = Path(root_path("data/training/labels.jsonl"))
TEST_SET = Path(root_path("data/training/kw_testset.json"))
TOP_N = 8


def _load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_testset() -> list:
    """读取独立测试集（data/training/kw_testset.json，27 篇全新文章，不参与训练）。

    Returns:
        [{doc_id,title,text,gold,related}, ...]
    """
    if not TEST_SET.exists():
        return []
    return json.loads(TEST_SET.read_text(encoding="utf-8"))


def _model_meta() -> dict:
    """读取正式模型元数据（训练样本数 / 校验指标），用于展示。"""
    meta_path = root_path("data/models/kw_rerank.json")
    if not meta_path.exists():
        return {"ready": False}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = meta.get("metrics", meta)
        return {
            "ready": True,
            "n_samples": meta.get("n_samples") or metrics.get("n_samples"),
            "val_spearman": metrics.get("val_spearman"),
            "val_class_acc": metrics.get("val_class_acc"),
            "trained_at": meta.get("trained_at"),
        }
    except Exception:
        return {"ready": False}


def run_eval(top_n: int = TOP_N) -> dict:
    """执行完整评估，返回结构化结果（JSON 可序列化）。

    Returns:
        {summary:{train_size,test_size,n_sci,n_lit,model_metrics},
         model:{...指标}, rule:{...指标},
         details:[{title,gold,top,hit_gold,n_gold,p8_gold,p8_rel,hit1}],
         conclusion:{verdict,compare}}
    """
    test_rows = build_testset()
    if not test_rows:
        return {"ok": False, "error": "独立测试集不存在，请先运行 tools._build_independent_testset 生成"}

    train_size = len(_load_jsonl(LBL))

    # 使用正式部署模型（非临时重训模型）
    formal = kw._load_rerank()
    saved_rerank, saved_enabled = kw._RERANK, kw._RERANK_ENABLED

    def run(use_model: bool):
        kw._RERANK = formal if use_model else None
        kw._RERANK_ENABLED = use_model
        out = []
        for tr in test_rows:
            # 与线上一致：传入文章标题（去掉文件名前缀），标题词特征才会生效
            title = tr["title"].split("_", 1)[-1]
            res = kw.extract_keywords_full(tr["text"], top_n=top_n, title=title)
            tops = [k["word"] for k in res["keywords"]]
            gold = set(tr["gold"]); rel = set(tr["related"])
            hit_gold = [w for w in tops if w in gold]
            out.append({
                "title": tr["title"],
                "top": tops,
                "gold": tr["gold"],
                "hit_gold": hit_gold,
                "n_gold": len(gold),
                "p8_gold": len(hit_gold) / top_n,
                "p8_rel": sum(1 for w in tops if w in gold or w in rel) / top_n,
                "hit1": bool(tops) and tops[0] in gold,
            })
        return out

    try:
        model_rows = run(True)
        rule_rows = run(False)
    finally:
        kw._RERANK, kw._RERANK_ENABLED = saved_rerank, saved_enabled

    def summ(rows_):
        n = len(rows_)
        return {
            "count": n,
            "hit1": sum(1 for r in rows_ if r["hit1"]) / n,
            "any_hit": sum(1 for r in rows_ if r["hit_gold"]) / n,
            "p8_gold": sum(r["p8_gold"] for r in rows_) / n,
            "p8_rel": sum(r["p8_rel"] for r in rows_) / n,
            "recall": sum(len(r["hit_gold"]) for r in rows_) / max(1, sum(r["n_gold"] for r in rows_)),
        }

    m, r_ = summ(model_rows), summ(rule_rows)

    if m["recall"] >= 0.5 and m["hit1"] >= 0.4:
        verdict = "效果良好：半数以上关键中心词能进入 Top%s，首名命中率高。" % top_n
    elif m["recall"] >= 0.3:
        verdict = "效果中等：关键词能部分进入 Top，但仍有提升空间（建议精调标注/语义门槛）。"
    else:
        verdict = "效果偏弱：关键中心词进入 Top 的比例偏低，需重点排查候选生成/去重阈值。"
    compare = ("模型版优于规则版（P@8 %.2f > %.2f），有监督重排带来提升。" % (m["p8_gold"], r_["p8_gold"])
               if m["p8_gold"] >= r_["p8_gold"]
               else "规则版反超模型版（P@8 %.2f < %.2f），标注/特征仍有偏差，需校正。" % (m["p8_gold"], r_["p8_gold"]))

    meta = _model_meta()
    return {
        "ok": True,
        "summary": {
            "train_size": train_size,
            "test_size": len(test_rows),
            "n_sci": sum(1 for t in test_rows if t["title"].startswith("w_")),
            "n_lit": sum(1 for t in test_rows if t["title"].startswith("lit_")),
            "top_n": top_n,
            "model_metrics": {
                "n_samples": meta.get("n_samples"),
                "val_spearman": meta.get("val_spearman"),
                "val_class_acc": meta.get("val_class_acc"),
                "trained_at": meta.get("trained_at"),
            },
        },
        "model": m, "rule": r_,
        "details": model_rows,
        "conclusion": {"verdict": verdict, "compare": compare},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--top-n", type=int, default=TOP_N)
    args = ap.parse_args()

    result = run_eval(top_n=args.top_n)
    if result.get("ok") is False:
        print(result.get("error", "评估失败"))
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return 0

    s = result["summary"]
    print("训练集=%d 测试集=%d（科普%d 文学%d）Top%d" % (s["train_size"], s["test_size"], s["n_sci"], s["n_lit"], s["top_n"]))
    mm = s["model_metrics"]
    print("正式模型: 样本=%s Spearman=%s 档位命中=%s"
          % (mm.get("n_samples"), mm.get("val_spearman"), mm.get("val_class_acc")))
    for key, label in (("model", "重排模型版"), ("rule", "规则版基线")):
        d = result[key]
        print("[%s] Hit@1=%.0f%% 篇命中=%.0f%% P@8(关键)=%.2f P@8(相关)=%.2f Gold召回=%.2f"
              % (label, d["hit1"] * 100, d["any_hit"] * 100, d["p8_gold"], d["p8_rel"], d["recall"]))
    print("判断: %s" % result["conclusion"]["verdict"])
    print("对比: %s" % result["conclusion"]["compare"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
