# -*- coding: utf-8 -*-
"""为 data/huaci/{学科}/*.jsonl 的每道题生成划词候选池，供 AI/人工标注"应划/不划"。

输出：data/huaci/{学科}/pool.json
  [{qid, subject, question, options, candidates:[{kw,type,rank,...}]}]

用法：py -3.12 -m tools.huaci_gen_pool
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import root_path  # noqa: E402
from modules.semantic import huaci_candidates as hc  # noqa: E402

BASE = Path(root_path("data/huaci"))


def main() -> int:
    total = 0
    for sub_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        qfiles = sorted(sub_dir.glob("ceval_*.jsonl"))
        if not qfiles:
            continue
        rows = []
        for qf in qfiles:
            for line in qf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                cands = hc.generate_question_candidates(
                    rec["question"], rec.get("options"), subject=rec["subject"]
                )
                rows.append({
                    "qid": rec["qid"],
                    "subject": rec["subject"],
                    "question": rec["question"],
                    "options": rec.get("options", []),
                    "candidates": cands,
                })
        out_path = sub_dir / "pool.json"
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        n_cand = sum(len(r["candidates"]) for r in rows)
        total += len(rows)
        print("%s: %d 题 / %d 候选 -> %s" % (sub_dir.name, len(rows), n_cand, out_path))
    print("总计 %d 题" % total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
