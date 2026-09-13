# -*- coding: utf-8 -*-
"""划词成果临时演示（不集成进项目，用完可删）。

用法：
  py demo_huaci.py              # 演示：抽 4 道真实题（数学+语文）看划词候选
  py demo_huaci.py "题目文本"    # 自定义：把你粘贴的题干/选项当作一道题看候选

说明：输出每道题的候选词 + 类型（data=数据类/logic=逻辑类/concept=概念类）
与是否来自选项，直观验证"划词候选生成"这个成果。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.semantic import huaci_candidates as hc  # noqa: E402

TYPE_CN = {"data": "数据类", "logic": "逻辑类", "concept": "概念类"}


def show(question: str, options=None, title: str = "题目") -> None:
    cands = hc.generate_question_candidates(question, options or [])
    print("-" * 62)
    print(title + "：", question[:80])
    if options:
        print("选项：", " | ".join(o[:24] for o in options[:4]))
    print("划词候选（共 %d 个）：" % len(cands))
    for c in cands:
        tags = []
        tags.append(TYPE_CN.get(c["type"], c["type"]))
        if c.get("is_formula"):
            tags.append("公式")
        if c.get("has_unit"):
            tags.append("带单位")
        if c.get("in_options"):
            tags.append("来自选项")
        print("  %2d. %-18s [%s] 频次=%d" % (c["rank"], c["kw"], "、".join(tags), c["freq"]))
    print()


def demo() -> None:
    base = Path("data/huaci")
    # 抽几道有代表性的题（数学公式题 / 语文知识题 / 语文阅读题）
    samples = []
    for subj, n in (("数学", 2), ("语文", 2)):
        pool = json.loads((base / subj / "pool.json").read_text(encoding="utf-8"))
        random.Random(7).shuffle(pool)
        for r in pool[:n]:
            samples.append(r)
    for r in samples:
        show(r["question"], r.get("options"), title="[%s] " % r["subject"] + r["qid"])


def main() -> int:
    if len(sys.argv) > 1:
        show(sys.argv[1], title="自定义题目")
        return 0
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
