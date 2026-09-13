# -*- coding: utf-8 -*-
"""构建"审题题库"：从 CJEval 标准化库筛选"无法直接写答案"的题型，并生成标准划词。

筛选（宁缺毋滥，保证"纯审题"定位）：
- 物理：计算题、简答题
- 语文：现代文阅读、诗歌鉴赏
排除：单选/多选/填空/选择（这些有确定答案，仍走答题）

标准划词：用现有 huaci_predict.predict_question 为每题生成"该题应划关键词"，
写进 keywords 字段（入库时可改、可编辑；审题时只读已存信息）。

产出：data/readingbank/{物理,语文}.jsonl
用法：py -3.12 -m tools.build_reading_bank
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import root_path
from modules.semantic import huaci_predict

SRC = Path(root_path("data/questions_cjeval"))
OUT = Path(root_path("data/readingbank"))
TOP_K = 6
# 各学科的"纯审题"题型白名单
KEEP_TYPE = {
    "物理": {"计算题", "简答题"},
    "语文": {"现代文阅读", "诗歌鉴赏"},
}
SUBJ_MAP = {"初中物理": "物理", "初中语文": "语文"}
_WS = re.compile(r"\s+")
_CHAR_OR_HAN = re.compile(r"[\u4e00-\u9fa5]")


def _clean(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for src_name, bank_name in (("初中物理", "物理"), ("初中语文", "语文")):
        src = SRC / f"{src_name}.jsonl"
        if not src.exists():
            print(f"[跳过] 无 {src}")
            continue
        keep = KEEP_TYPE[bank_name]
        recs, kept, skipped = [], 0, 0
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            if it.get("ques_type") not in keep:
                skipped += 1
                continue
            question = _clean(it.get("question", ""))
            if len(question) < 10:
                skipped += 1
                continue
            # 生成该题"应划关键词"，作为标准划词（入库后人工可改）
            try:
                preds = huaci_predict.predict_question(question, it.get("options") or [],
                                                       subject=bank_name, top_k=TOP_K)
                keywords = [p.get("kw", "") for p in preds if p.get("kw")]
            except Exception:
                keywords = []
            recs.append({
                "id": f"rb_{bank_name}_{kept:05d}",
                "subject": bank_name,
                "source": "CJEval",
                "ques_type": it.get("ques_type", ""),
                "difficulty": it.get("difficulty", ""),
                "question": question,
                "options": it.get("options") or [],
                "keywords": keywords,          # 标准划词：入库时生成，可人工修改
                "keywords_version": "auto",
                "answer_text": it.get("answer_text", ""),
                "knowledges": it.get("knowledges") or [],
            })
            kept += 1
        out = OUT / f"{bank_name}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        withkw = sum(1 for r in recs if r["keywords"])
        print(f"{bank_name}: 审题题库 {len(recs)} 题（跳过 {skipped}）| 已生成标准划词 {withkw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())