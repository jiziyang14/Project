"""文章关键词标注与训练数据服务（仅供学习伴侣本地使用）。

提供：
- 语料库（data/training/articles.jsonl）：存放待标注/已标注的文章。
- 标注底稿（data/training/labels.jsonl）：每篇文章的候选词 + 特征 + 人工/AI 标分。
- 候选词导出：复用 semantic.keywords.generate_candidates 流水线。
- AI 预标：通过 OpenAI 兼容接口（LLM_BASE_URL/LLM_API_KEY/LLM_MODEL）为候选词打分，
  人工仅复核修改；未配置大模型时该功能优雅降级为"请先配置"提示。

label 语义：0=无关/噪声压掉，1=相关但靠后，2=关键中心词（必上屏）。
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.config import root_path
from core.logger import logger

_LOCK = threading.RLock()

ARTICLES_PATH = lambda: root_path("data/training/articles.jsonl")
LABELS_PATH = lambda: root_path("data/training/labels.jsonl")
RERANK_PKL = lambda: root_path("data/models/kw_rerank.pkl")
RERANK_JSON = lambda: root_path("data/models/kw_rerank.json")

# 候选词导出每篇数量（标注时看到的候选规模）
EXPORT_TOP_K = 18


# ── 读写辅助 ─────────────────────────────────
def _read_jsonl(path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("读取 %s 失败：%s", path, exc)
    return out


def _write_jsonl(path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def corpus() -> List[dict]:
    return _read_jsonl(ARTICLES_PATH())


def labels_map() -> Dict[str, dict]:
    return {row["doc_id"]: row for row in _read_jsonl(LABELS_PATH())}


def _labels_rows() -> List[dict]:
    return list(labels_map().values())


# ── 核心操作 ─────────────────────────────────
def add_article(title: str, text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("文章正文不能为空")
    doc = {
        "doc_id": str(uuid.uuid4()),
        "title": (title or text[:20]).strip(),
        "text": text,
        "source": "manual",
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    with _LOCK:
        _append_jsonl(ARTICLES_PATH(), doc)
    return {"doc_id": doc["doc_id"], "title": doc["title"]}


def _text_exists(text: str) -> bool:
    return any(d.get("text", "").strip() == text.strip() for d in corpus())


def import_folder(texts_dir: str | None = None) -> dict:
    """批量导入指定文件夹（默认 data/texts/）下的所有 .txt，每个文件一篇文章。

    Returns:
        {"imported": 新增数, "skipped": 重复或空文件数}。
    """
    folder = Path(texts_dir) if texts_dir else root_path("data/texts")
    if not folder.exists() or not folder.is_dir():
        return {"imported": 0, "skipped": 0, "path": str(folder)}
    imported = skipped = 0
    for p in sorted(folder.glob("*.txt")):
        if p.name.startswith("_"):
            continue
        try:
            text = p.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("读取 %s 失败：%s", p.name, exc)
            skipped += 1
            continue
        if len(text) < 30 or _text_exists(text):
            skipped += 1
            continue
        with _LOCK:
            add_article(p.stem, text)
        imported += 1
    return {"imported": imported, "skipped": skipped, "path": str(folder)}


def import_buffers(files) -> dict:
    """把上传的多个 .txt（内存内容）批量加入语料库。"""
    imported = skipped = 0
    for f in files:
        try:
            text = f.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            skipped += 1
            continue
        if len(text) < 30 or _text_exists(text):
            skipped += 1
            continue
        with _LOCK:
            add_article(Path(f.filename or "未命名").stem, text)
        imported += 1
    return {"imported": imported, "skipped": skipped}


def export_pending() -> dict:
    """为语料库中尚未导出候选词的文章生成候选 + 特征，写入标注底稿。"""
    from modules.semantic import keywords as kw

    with _LOCK:
        lm = labels_map()
        new_rows = _labels_rows_by(lm)
        exported, skipped_empty, was = 0, 0, 0
        for doc in corpus():
            if doc["doc_id"] in lm:
                continue  # 已导出
            try:
                cands = kw.generate_candidates(doc["text"], top_k=EXPORT_TOP_K)
            except Exception as exc:
                logger.warning("候选词导出失败 %s：%s", doc["doc_id"], exc)
                was += 1
                continue
            if not cands:
                skipped_empty += 1
                # 空候选也占位，下次不重复处理
            row = {
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "candidates": [dict(c, **{"label": None, "note": ""}) for c in cands],
            }
            new_rows.append(row)
            lm[doc["doc_id"]] = row
            exported += 1
        _write_jsonl(LABELS_PATH(), new_rows)
    return {"exported": exported, "skipped_empty": skipped_empty, "failed": was}


def _labels_rows_by(lm: Dict[str, dict]) -> List[dict]:
    return [lm[k] for k in lm] if lm else []


def list_items() -> dict:
    docs = corpus()
    lm = labels_map()
    items = []
    labeled_docs = 0
    for doc in docs:
        row = lm.get(doc["doc_id"])
        cands = (row or {}).get("candidates", [])
        is_labeled = bool(cands) and any(c.get("label") is not None for c in cands)
        if is_labeled:
            labeled_docs += 1
        items.append({
            "doc_id": doc["doc_id"],
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "exported": bool(cands),
            "labeled": is_labeled,
            "candidates": cands,
        })
    return {"items": items, "counts": {"total": len(docs), "labeled": labeled_docs, "need_label": len(docs) - labeled_docs}}


def save_labels(doc_id: str, candidates: List[dict]) -> dict:
    cleaned = []
    for c in candidates or []:
        label = c.get("label")
        if label not in (0, 1, 2, "0", "1", "2"):
            label = None
        else:
            label = int(label)
        cleaned.append({
            "kw": c["kw"],
            "basis": c.get("basis", ""),
            "frequency": c.get("frequency", 0),
            "coverage": c.get("coverage", 0),
            "stat_score": c.get("stat_score", 0),
            "stat_norm": c.get("stat_norm", 0),
            "semantic": c.get("semantic", 0),
            "length": c.get("length", len(c.get("kw", ""))),
            "is_subsumed": c.get("is_subsumed", 0),
            "rule_conf": c.get("rule_conf", 0),
            "label": label,
            "note": (c.get("note") or "").strip(),
        })
    with _LOCK:
        rows = _labels_rows()
        row = next((r for r in rows if r["doc_id"] == doc_id), None)
        if row is None:
            # 允许对未导出的文章直接保存（空候选兜底，避免前端状态丢失）
            doc = next((d for d in corpus() if d["doc_id"] == doc_id), None)
            if doc is None:
                raise ValueError("文档不存在")
            row = {"doc_id": doc_id, "title": doc.get("title", ""), "text": doc.get("text", ""), "candidates": []}
            rows.append(row)
        row["candidates"] = _fill_user_features(row.get("text", ""), cleaned)
        _write_jsonl(LABELS_PATH(), rows)
    return {"doc_id": doc_id, "saved": len(cleaned)}


def _fill_user_features(text: str, cleaned: List[dict]) -> List[dict]:
    """为「划词标记」的词补全特征（流水线候选已有特征，无需处理）。

    划词词没有统计/语义分，这里用朴素子串计数给出词频、覆盖率和长度等，
    使它们也能作为正样本参与重排训练。
    """
    n = max(1, len(text or ""))
    pool = [x["kw"] for x in cleaned]
    for c in cleaned:
        if c.get("basis") != "划词标记":
            continue
        kw = c["kw"]
        freq = max(1, (text or "").count(kw))
        c["frequency"] = freq
        c["coverage"] = round(freq / n, 4)
        c["length"] = len(kw)
        c["is_subsumed"] = 1 if any(kw != o and kw in o for o in pool) else 0
        c["stat_score"] = c.get("stat_score", 0)
        c["stat_norm"] = c.get("stat_norm", 0)
        c["semantic"] = c.get("semantic", 0)
        c["rule_conf"] = c.get("rule_conf", 0)
    return cleaned


# ── AI 预标（OpenAI 兼容接口） ─────────────────
def _llm_config():
    base = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()
    if not base or not key:
        return None
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    return {"url": url, "key": key, "model": model}


def _llm_chat(prompt: str, max_tokens: int = 1200) -> str:
    cfg = _llm_config()
    if not cfg:
        raise RuntimeError("未配置大模型服务（LLM_BASE_URL / LLM_API_KEY），请设置后重试")
    body = json.dumps({
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        cfg["url"], data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['key']}"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _extract_json(text: str):
    """从 LLM 输出中稳健解析 JSON 数组/对象（容忍前后缀文字与围栏）。"""
    start = min((idx for idx in (text.find("["), text.find("{")) if idx >= 0), default=-1)
    if start < 0:
        raise ValueError("AI 未返回 JSON")
    end = max(text.rfind("]"), text.rfind("}"))
    return json.loads(text[start:end + 1])


LABEL_HINT = "0=无关/噪声(压掉) 1=相关但次要 2=关键中心词(必上屏)"


def ai_prelabel(doc_id: str) -> dict:
    row = labels_map().get(doc_id)
    if not row or not row.get("candidates"):
        raise ValueError("该文章尚未导出候选词，请先导出")
    cands = row["candidates"]
    pool = "\n".join(
        f"{i+1}. {c['kw']}（频次={c.get('frequency')}, 语义贴合={c.get('semantic'):.3f}, 规则分={c.get('rule_conf'):.3f}）"
        for i, c in enumerate(cands)
    )
    prompt = (
        "你是文章关键词标注助手。下面是一篇文章与候选词列表。\n"
        f"【文章】\n{row.get('text','')}\n"
        f"【候选词】\n{pool}\n"
        f"请给每个候选词按语义标分：{LABEL_HINT}。\n"
        "严格仅返回 JSON 数组，元素为 {\"kw\":\"词\",\"label\":0或1或2,\"reason\":\"一句话理由\"}，"
        "KW 必须与候选词完全一致。不要输出任何其它文字。"
    )
    try:
        raw = _llm_chat(prompt)
        parsed = _extract_json(raw)
    except Exception as exc:
        logger.warning("AI 预标失败：%s", exc)
        raise RuntimeError(f"AI 预标失败：{exc}")
    by_kw = {p.get("kw"): p for p in parsed if isinstance(p, dict)}
    out = []
    for c in cands:
        p = by_kw.get(c["kw"], {})
        lbl = p.get("label")
        out.append({
            "kw": c["kw"],
            "label": lbl if lbl in (0, 1, 2) else None,
            "reason": str(p.get("reason", "") or ""),
            "ai": True,
        })
    return {"items": out, "hint": LABEL_HINT}


# ── 划词标注（读题划词） ───────────────────────────────────────────
# 与文章关键词共用同一套 0/1/2 标注语义，但题源（data/huaci/{学科}/pool.json）
# 与标注（data/huaci/{学科}/labels.jsonl）分开存放，互不干扰、可独立导出训练。
HUACI_SUBJECTS = ["数学", "语文"]
HUACI_LABEL_HINT = "0=不划(噪音压掉) 1=可划但次要 2=必划(读题重点划取)"


def _huaci_pool(subj: str) -> List[dict]:
    path = root_path(f"data/huaci/{subj}/pool.json")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("读取划词题源 %s 失败：%s", path, exc)
        return []


def _huaci_labels_map(subj: str) -> Dict[str, dict]:
    return {r["qid"]: r for r in _read_jsonl(root_path(f"data/huaci/{subj}/labels.jsonl"))}


def _huaci_labels_rows(subj: str) -> List[dict]:
    return list(_huaci_labels_map(subj).values())


def huaci_status() -> dict:
    """每个学科的题量 / 已标注 / 待标注。"""
    out = {}
    total_q = labeled_q = 0
    for subj in HUACI_SUBJECTS:
        pool = _huaci_pool(subj)
        lm = _huaci_labels_map(subj)
        labeled = sum(
            1 for q in pool
            if q["qid"] in lm and any(c.get("label") is not None for c in lm[q["qid"]].get("candidates", []))
        )
        out[subj] = {"total": len(pool), "labeled": labeled, "need_label": max(0, len(pool) - labeled)}
        total_q += len(pool)
        labeled_q += labeled
    return {"subjects": out, "total": total_q, "labeled": labeled_q, "hint": HUACI_LABEL_HINT}


def huaci_items(subj: str) -> dict:
    """把 pool.json 的候选与已存标注合并，供标注页逐题浏览。"""
    pool = _huaci_pool(subj)
    lm = _huaci_labels_map(subj)
    items = []
    labeled_q = 0
    for q in pool:
        row = lm.get(q["qid"])
        saved = {c["kw"]: c for c in (row or {}).get("candidates", [])}
        cands = []
        for c in q.get("candidates", []):
            s = saved.get(c["kw"], {})
            cands.append(dict(c, **{
                "label": s.get("label"),
                "note": s.get("note", ""),
                "reason": s.get("reason", ""),
                "ai": s.get("ai", False),
            }))
        is_labeled = any(c["label"] is not None for c in cands)
        if is_labeled:
            labeled_q += 1
        items.append({
            "qid": q["qid"], "subject": q.get("subject", subj),
            "question": q.get("question", ""), "options": q.get("options", []),
            "candidates": cands, "labeled": is_labeled,
        })
    return {
        "items": items,
        "counts": {"total": len(items), "labeled": labeled_q, "need_label": max(0, len(items) - labeled_q)},
    }


def huaci_save(subj: str, qid: str, candidates: List[dict]) -> dict:
    cleaned = []
    for c in candidates or []:
        label = c.get("label")
        if label not in (0, 1, 2, "0", "1", "2"):
            label = None
        else:
            label = int(label)
        cleaned.append({
            "kw": c.get("kw", ""),
            "type": c.get("type", ""),
            "length": c.get("length", len(c.get("kw", ""))),
            "freq": c.get("freq", c.get("frequency", 0)),
            "first_pos_norm": c.get("first_pos_norm", 1.0),
            "in_options": c.get("in_options", 0),
            "is_formula": c.get("is_formula", 0),
            "is_number": c.get("is_number", 0),
            "has_unit": c.get("has_unit", 0),
            "is_kaodian": c.get("is_kaodian", 0),
            "label": label,
            "note": (c.get("note") or "").strip(),
            "reason": (c.get("reason") or "").strip(),
            "ai": bool(c.get("ai")),
        })
    path = root_path(f"data/huaci/{subj}/labels.jsonl")
    with _LOCK:
        rows = _huaci_labels_rows(subj)
        row = next((r for r in rows if r["qid"] == qid), None)
        if row is None:
            row = {"qid": qid, "subject": subj, "candidates": []}
            rows.append(row)
        row["candidates"] = cleaned
        row["saved_at"] = datetime.now().isoformat(timespec="seconds")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(path, rows)
    return {"qid": qid, "saved": len(cleaned)}


def huaci_export(subj: str) -> dict:
    """把已标注的划词样本导出为训练数据（供后续训练划词模型）。"""
    pool = {q["qid"]: q for q in _huaci_pool(subj)}
    rows = []
    labeled_q = 0
    for qid, row in _huaci_labels_map(subj).items():
        q = pool.get(qid)
        if not q:
            continue
        labeled = [c for c in row.get("candidates", []) if c.get("label") is not None]
        if not labeled:
            continue
        labeled_q += 1
        for c in labeled:
            rows.append({
                "source": "huaci", "subject": subj, "qid": qid,
                "question": q.get("question", ""), "options": q.get("options", []),
                "kw": c["kw"], "label": c["label"], "note": c.get("note", ""),
                "type": c.get("type", ""), "length": c.get("length", len(c["kw"])),
                "freq": c.get("freq", 0), "first_pos_norm": c.get("first_pos_norm", 1.0),
                "in_options": c.get("in_options", 0), "is_formula": c.get("is_formula", 0),
                "is_number": c.get("is_number", 0), "has_unit": c.get("has_unit", 0),
                "is_kaodian": c.get("is_kaodian", 0),
            })
    out_path = root_path(f"data/huaci/{subj}/train_samples.jsonl")
    with _LOCK:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_path, rows)
    return {"subject": subj, "labeled_questions": labeled_q, "samples": len(rows), "path": str(out_path)}


def huaci_eval(subj: str = "数学") -> dict:
    """一键评估划词模型：在已标注样本上做分层留出评估（0/1/2 分类）。

    train_samples.jsonl 由 huaci_export 产出，且与推理共用同一套特征
    （modules.semantic.huaci_candidates.huaci_to_features），避免口径漂移。
    label 仅作评估目标，不参与特征。留出的 20% 样本不参与本次训练。
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split

    from modules.semantic import huaci_candidates as hc

    path = root_path(f"data/huaci/{subj}/train_samples.jsonl")
    if not path.exists():
        return {"ok": False, "error": f"「{subj}」无训练样本，请先在读题划词标注页点「导出划词训练数据」"}
    rows = [r for r in _read_jsonl(path) if r.get("label") in (0, 1, 2)]
    if len(rows) < 20:
        return {"ok": False, "error": f"「{subj}」标注样本不足（{len(rows)} < 20）"}

    X = np.array([hc.huaci_to_features(r) for r in rows], dtype=np.float64)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = GradientBoostingClassifier(
        n_estimators=180, max_depth=3, min_samples_leaf=8, learning_rate=0.08,
        subsample=0.85, random_state=42,
    )
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xva)

    names = ["不划", "可划", "必划"]  # 0/1/2
    rep = classification_report(yva, yp, labels=[0, 1, 2], target_names=names,
                                output_dict=True, zero_division=0)
    per_class = {
        k: {kk: round(v[kk], 4) for kk in ("precision", "recall", "f1-score", "support")}
        for k, v in rep.items() if k in names
    }
    imp = sorted(zip(hc.huaci_feature_names(), clf.feature_importances_), key=lambda x: -x[1])[:5]

    return {
        "ok": True, "subject": subj, "n_samples": len(rows),
        "test_size": int(len(yva)),
        "accuracy": round(float(accuracy_score(yva, yp)), 4),
        "f1_weighted": round(float(f1_score(yva, yp, average="weighted")), 4),
        "f1_macro": round(float(f1_score(yva, yp, average="macro")), 4),
        "per_class": per_class,
        "top_features": [[str(f), round(float(i), 4)] for f, i in imp],
        "hint": "独立留出 20% 样本评估（不参与本次训练），衡量模型对「该不该划」0/1/2 的分类能力。",
    }


def rerank_status() -> dict:
    pkl = RERANK_PKL()
    meta = {}
    if RERANK_JSON().exists():
        try:
            meta = json.loads(RERANK_JSON().read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return {"ready": pkl.exists(), "enabled": True, **(meta or {})}


# ── FastAPI 路由 ──────────────────────────────
from fastapi import APIRouter, File, UploadFile

router = APIRouter(prefix="/api/train", tags=["关键词训练"])


@router.get("/status")
def _status():
    with _LOCK:
        docs = corpus()
        lm = labels_map()
    labeled = sum(1 for r in lm.values() if any(c.get("label") is not None for c in r.get("candidates", [])))
    try:
        return {
            "articles": len(docs),
            "exported": len(lm),
            "labeled": labeled,
            "rerank": rerank_status(),
        }
    except Exception:
        raise


@router.post("/article")
def _add_article(payload: dict):
    try:
        return add_article(payload.get("title", ""), payload.get("text", ""))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/import-folder")
def _import_folder(payload: dict | None = None):
    """扫描 data/texts/ 下所有 .txt 批量加入语料库（每个文件一篇文章）。"""
    try:
        res = import_folder((payload or {}).get("path") or None)
        return {"ok": True, **res}
    except Exception as exc:
        logger.error("文件夹导入失败：%s", exc)
        return {"ok": False, "error": f"导入失败：{exc}"}


@router.post("/import-files")
async def _import_files(files: list[UploadFile] = File(...)):
    """上传多个 .txt 文件批量加入语料库。"""
    try:
        res = import_buffers(files)
        return {"ok": True, **res}
    except Exception as exc:
        logger.error("文件导入失败：%s", exc)
        return {"ok": False, "error": f"导入失败：{exc}"}


@router.get("/articles")
def _articles():
    return {"articles": [{"doc_id": d["doc_id"], "title": d.get("title", "")} for d in corpus()]}


@router.post("/export")
def _export():
    try:
        res = export_pending()
        return {"ok": True, **res}
    except Exception as exc:
        logger.error("导出候选词失败：%s", exc)
        return {"ok": False, "error": f"导出失败：{exc}"}


@router.get("/items")
def _items():
    return list_items()


@router.post("/label")
def _label(payload: dict):
    try:
        res = save_labels(payload["doc_id"], payload.get("candidates", []))
        return {"ok": True, **res}
    except (ValueError, KeyError) as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/ai")
def _ai(payload: dict):
    try:
        res = ai_prelabel(payload["doc_id"])
        return {"ok": True, **res}
    except (ValueError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/rerank-run")
def _rerank_run():
    """在进程内训练重排器（数据量小，秒级完成）。"""
    try:
        from tools.train_rerank import train

        metrics = train(str(LABELS_PATH()), str(RERANK_PKL()), max_samples=0)
        return {"ok": True, "metrics": metrics}
    except Exception as exc:
        logger.error("重排器训练失败：%s", exc)
        return {"ok": False, "error": f"训练失败：{exc}"}


# ── 划词标注 API（复用 /api/train 路径，通过 mode 参数区分） ──────────
@router.get("/huaci-status")
def _huaci_status():
    try:
        return huaci_status()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/huaci-items")
def _huaci_items(subj: str = "数学"):
    try:
        return huaci_items(subj)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/huaci-save")
def _huaci_save(payload: dict):
    try:
        res = huaci_save(payload["subj"], payload["qid"], payload.get("candidates", []))
        return {"ok": True, **res}
    except (ValueError, KeyError) as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/huaci-export")
def _huaci_export(payload: dict):
    try:
        res = huaci_export(payload.get("subj", "数学"))
        return {"ok": True, **res}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/huaci-eval")
def _huaci_eval(payload: dict):
    """一键评估划词模型效果（区分于文章关键词的 /eval）。"""
    try:
        return huaci_eval(payload.get("subj", "数学"))
    except Exception as exc:
        logger.error("划词评估失败：%s", exc)
        return {"ok": False, "error": f"评估失败：{exc}"}


@router.post("/eval")
def _eval_run():
    """一键评估：用独立测试集衡量关键词提取效果（不覆盖正式模型）。"""
    try:
        from tools.kw_eval import run_eval

        res = run_eval(top_n=8)
        return {"ok": True, **res}
    except Exception as exc:
        logger.error("评估执行失败：%s", exc)
        return {"ok": False, "error": f"评估失败：{exc}"}