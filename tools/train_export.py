"""导出文章关键词标注底稿（CLI）。

用法：
    python -m tools.train_export                # 对语料库现有文章导出候选词（复用生成流水线）
    python -m tools.train_export --seed         # 先把题库 data/questions.json 作为文章补进语料库再导出
    python -m tools.train_export --seed --max 50  # 只取题库前 50 题（用于快速起步）

导出结果写入 data/training/labels.jsonl（候选词 + 全量特征，label 待标注），
供后端网页标注或直接供 tools/train_rerank.py 训练。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.config import root_path


def seed_from_questions(max_items: int = 0) -> int:
    """把题库题目作为文章补进语料库，返回新增条数。"""
    from backend.training import add_article, corpus

    qf = root_path("data/questions.json")
    if not qf.exists():
        print("[提示] 未找到 data/questions.json，跳过题库播种")
        return 0
    questions = json.loads(qf.read_text(encoding="utf-8"))
    existing = {d["doc_id"] for d in corpus()}
    # 题库无独立 doc 去重键，用标题近似；重复添加由调用方控制
    added = 0
    for q in questions[:max_items] if max_items else questions:
        text = (q.get("question_text") or "").strip()
        if not text or len(text) < 30:
            continue
        add_article(q.get("knowledge_tags") and q["knowledge_tags"][0] or "题库题", text)
        added += 1
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description="导出文章关键词标注底稿")
    ap.add_argument("--seed", action="store_true", help="把题库 data/questions.json 播种进语料库")
    ap.add_argument("--max", type=int, default=0, help="播种或导出的条数上限")
    args = ap.parse_args()

    if args.seed:
        print(f"已完成题库播种：新增 {seed_from_questions(args.max)} 篇文章")

    from backend.training import export_pending

    res = export_pending()
    print(f"导出完成：新增候选底稿 {res['exported']} 篇，{res['skipped_empty']} 篇未提取到候选，{res['failed']} 篇失败")
    if res["exported"] == 0:
        print("[提示] 语料库为空或全部已导出。先在网页「关键词训练」页添加文章，或运行 --seed。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())