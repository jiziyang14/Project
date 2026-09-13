# -*- coding: utf-8 -*-
"""导入 CJEval（初中物理/语文）→ 项目标准题库格式。

CJEval: https://github.com/SmileWHC/CJEval（学术基准，仅供研究用，本项目为高中探究汇报）
原始文件：data/external/CJEval_raw/{train,valid,test}_{初中物理,初中语文}.json（JSONL）
输出　　：data/questions_cjeval/{初中物理,初中语文}.jsonl（统一字段）

统一字段：{id, source, split, subject, ques_type, difficulty, question, options,
            answer(列表), answer_text, analyze, knowledges[]}
选项从内容里的"选项: A.… B.…"内联块提取并剥离；无选项则 options=[]。
用法：py -3.12 -m tools.import_cjeval
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import root_path

RAW = Path(root_path("data/external/CJEval_raw"))
OUT = Path(root_path("data/questions_cjeval"))
SUBJECTS = ["初中物理", "初中语文"]
SPLITS = ["train", "valid", "test"]

_OPT_HEAD = re.compile(r"选项\s*[:：;；]?\s*(?=[A-H]\.)", re.S)
_OPT_MARK = re.compile(r"[A-H]\.\s*")
_LEAD = re.compile(r"^\s*题目内容\s*[:：]?\s*")


def _strip(c: str) -> str:
    c = (_LEAD.sub("", c or "")).strip()
    return c


def _split_question_options(c):
    """把内联选项块从题干剥离。返回 (question, options[])。"""
    c = _strip(_flatten_content(c))
    m = _OPT_HEAD.search(c)
    if not m:
        return c, []
    q_text = c[: m.start()].rstrip().rstrip("（").rstrip("(").rstrip("）").rstrip(")").strip()
    opts_text = c[m.end():]
    parts = [p for p in _OPT_MARK.split(opts_text) if p and p.strip()]
    # 去掉尾部换行/多余符号
    options = [p.strip() for p in parts]
    return q_text, options


def _flatten_content(c):
    """把 ques_content（str / dict / 字符串化的dict）展平成题干文本。"""
    if isinstance(c, str):
        s = c.strip()
        # 部分记录 ques_content 是"字符串化的字典"（Python 字面量），需还原后递归
        if s.startswith("{"):
            try:
                import ast
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    return _flatten_content(obj)
            except Exception:
                pass
        return c
    if isinstance(c, dict):
        # 现代文阅读：{'题目','段落','问题'} 或 {'ques_type','ques_content'} 或 {'问题',...}
        if "段落" in c and "问题" in c:
            title = str(c.get("题目", "") or "").strip()
            paras = "\n".join(p for p in (c.get("段落") or []) if p)
            qs = "\n".join("问题：" + str(x) for x in (c.get("问题") or []))
            return "\n\n".join(x for x in (title, paras, qs) if x)
        for key in ("ques_content", "题目", "问题", "text", "content"):
            v = c.get(key)
            if v:
                return json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        return json.dumps(c, ensure_ascii=False)
    return json.dumps(c, ensure_ascii=False) if isinstance(c, (list, tuple)) else str(c)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for subj in SUBJECTS:
        recs = []
        for sp in SPLITS:
            src = RAW / f"{sp}_{subj}.json"
            if not src.exists():
                continue
            for line in src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                q, opts = _split_question_options(it.get("ques_content", ""))
                ans = it.get("ques_answer") or []
                recs.append({
                    "id": None,  # 由后端统一分配稳定 id
                    "source": "CJEval", "split": sp, "subject": subj,
                    "ques_type": it.get("ques_type", ""),
                    "difficulty": it.get("ques_difficulty", ""),
                    "question": q,
                    "options": opts,
                    "answer": [str(a) for a in ans],
                    "answer_text": "/".join(str(a) for a in ans),
                    "analyze": it.get("ques_analyze", ""),
                    "knowledges": it.get("ques_knowledges") or [],
                })
        out = OUT / f"{subj}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 统计
        n_opt = sum(1 for r in recs if r["options"])
        n_long = sum(1 for r in recs if len(r["question"]) >= 200)
        n_type = {}
        for r in recs:
            n_type[r["ques_type"]] = n_type.get(r["ques_type"], 0) + 1
        print(f"{subj}: 共 {len(recs)} 题 | 带选项 {n_opt} | 题干>=200字 {n_long}")
        print(f"   题型: {n_type}  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())