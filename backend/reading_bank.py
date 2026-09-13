"""审题题库（纯审题练习）后端服务。

题源：data/readingbank/{物理,语文}.jsonl
每题含 keywords（该题应划的标准划词，入库时自动生成、可人工修改）。
审题练习时前端读取已存 keywords 作对照标尺，本次会话不重新生成。
"""
from __future__ import annotations

import json
import random
import threading
from collections import Counter
from fastapi import APIRouter
from typing import List

from core.config import root_path
from core.logger import logger

router = APIRouter(prefix="/api/readingbank", tags=["审题题库"])

_LOCK = threading.RLock()
_BANK_PATH = lambda subj: root_path(f"data/readingbank/{subj}.jsonl")
_SUBJECTS = ("物理", "语文")


def _read(subject: str) -> List[dict]:
    rows = []
    path = _BANK_PATH(subject)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write(subject: str, rows: List[dict]) -> None:
    path = _BANK_PATH(subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@router.get("/status")
def _status():
    out = {}
    for s in _SUBJECTS:
        rows = _read(s)
        out[s] = {
            "total": len(rows),
            "with_keywords": sum(1 for r in rows if r.get("keywords")),
            "difficulties": sorted({r.get("difficulty", "") for r in rows if r.get("difficulty")}),
            "types": sorted({r.get("ques_type", "") for r in rows if r.get("ques_type")}),
        }
    return {"subjects": out}


@router.get("/list")
def _list(subject: str = "物理", offset: int = 0, limit: int = 20):
    rows = _read(subject)
    rows = rows[offset:offset + limit]
    return {
        "subject": subject,
        "items": [{
            "id": r["id"], "ques_type": r.get("ques_type", ""),
            "difficulty": r.get("difficulty", ""),
            "question": r.get("question", ""),
            "keywords": r.get("keywords") or [],
        } for r in rows],
    }


@router.get("/item")
def _item(subject: str = "物理", id: str = ""):
    rows = _read(subject)
    hit = next((r for r in rows if r["id"] == id), None)
    if not hit:
        return {"ok": False, "error": "未找到该题"}
    return {"ok": True, "question": hit}


@router.get("/random")
def _random(subject: str = "物理", difficulty: str = "", ques_type: str = "", seed: int | None = None):
    """按筛选条件抽一道可读的审题题（带标准划词）。"""
    rows = [r for r in _read(subject) if r.get("keywords")]
    if difficulty:
        rows = [r for r in rows if (r.get("difficulty") or "") == difficulty]
    if ques_type:
        rows = [r for r in rows if (r.get("ques_type") or "") == ques_type]
    if not rows:
        return {"ok": False, "error": f"「{subject}」审题题库在所选条件下无题"}
    rng = random.Random(seed) if seed is not None else random
    return {"ok": True, "question": rng.choice(rows)}


@router.post("/save-keywords")
def _save_keywords(payload: dict):
    subject = payload.get("subject", "物理")
    qid = payload.get("id", "")
    keywords = [str(k).strip() for k in (payload.get("keywords") or []) if str(k).strip()]
    with _LOCK:
        rows = _read(subject)
        hit = next((r for r in rows if r["id"] == qid), None)
        if not hit:
            return {"ok": False, "error": "未找到该题"}
        hit["keywords"] = keywords
        hit["keywords_version"] = "manual"
        _write(subject, rows)
    return {"ok": True, "id": qid, "keywords": keywords}