# -*- coding: utf-8 -*-
"""划词训练数据管道：抓取并清洗 C-Eval 数学/语文题目，落盘到 data/huaci/。

数据源：C-Eval（上海AI实验室，ceval/ceval-exam，CC-BY-NC-4.0）
  - 数学：middle_school_mathematics / high_school_mathematics / advanced_mathematics
  - 语文：high_school_chinese / chinese_language_and_literature
经 hf-mirror.com 镜像下载（国内可直连）。

输出（每个学科一个目录，JSONL，一题一行）：
  data/huaci/数学/ceval_初中数学.jsonl
  data/huaci/数学/ceval_高中数学.jsonl
  data/huaci/数学/ceval_高等数学.jsonl
  data/huaci/语文/ceval_高中语文.jsonl
  data/huaci/语文/ceval_中国语言文学.jsonl

每行字段：{source, subject, qid, question, options, answer, meta}

用法：py -3.12 -m tools.huaci_build_data
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import root_path  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = "ceval/ceval-exam"

# 学科名（目录）→ (落盘学科目录, 落盘文件名)
SUBJECTS = [
    ("middle_school_mathematics", "数学", "ceval_初中数学"),
    ("high_school_mathematics", "数学", "ceval_高中数学"),
    ("advanced_mathematics", "数学", "ceval_高等数学"),
    ("high_school_chinese", "语文", "ceval_高中语文"),
    ("chinese_language_and_literature", "语文", "ceval_中国语言文学"),
]

SPLITS = ["dev", "val", "test"]

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    """清洗题干：合并空白、去行首行尾、规范化标点。"""
    if not text:
        return ""
    text = text.replace("\u3000", " ").replace("\n", " ")
    text = _WS.sub(" ", text).strip()
    return text


def _download_and_dump(sub_dir: str, out_dir: Path, out_stem: str, subject_cn: str) -> dict:
    """下载一个学科的 dev/val/test parquet 并合并为 JSONL。

    用 requests 直接下载（镜像对 huggingface_hub 客户端的 HEAD 请求不稳定，
    改为流式 GET + 长超时，更稳）。
    """
    import requests

    import pandas as pd

    mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    session = requests.Session()

    rows = []
    for split in SPLITS:
        url = f"{mirror}/datasets/{REPO}/resolve/main/{sub_dir}/{split}-00000-of-00001.parquet"
        tmp = Path(root_path("data/huaci/_tmp.parquet"))
        # 优先复用已下载的本地缓存（避免镜像不稳定导致重复等待）
        cached = Path(root_path("data/huaci/_tmp_%s_%s.parquet" % (sub_dir, split)))
        src = cached if cached.exists() else tmp
        downloaded = cached.exists()
        if not downloaded:
            # 镜像间歇性超时，最多重试 6 次、间隔 3s
            for attempt in range(1, 7):
                try:
                    with session.get(url, timeout=120, stream=True) as resp:
                        resp.raise_for_status()
                        with open(tmp, "wb") as fh:
                            for chunk in resp.iter_content(65536):
                                if chunk:
                                    fh.write(chunk)
                    downloaded = True
                    break
                except Exception as exc:
                    print("  [retry %d/6] %s/%s：%s" % (attempt, sub_dir, split, str(exc)[:80]))
                    time.sleep(3)
        if not downloaded:
            print("  [skip] %s/%s 下载失败" % (sub_dir, split))
            continue
        try:
            df = pd.read_parquet(src)
        except Exception as exc:
            print("  [skip] %s/%s 读取失败：%s" % (sub_dir, split, str(exc)[:100]))
            continue
        for _, rec in df.iterrows():
            rec = {k: ("" if v is None else str(v)) for k, v in rec.items()}
            question = _clean(rec.get("question", ""))
            if not question:
                continue
            # 选项：A/B/C/D 依次拼接（存在的才加）
            options = []
            for opt in ("A", "B", "C", "D"):
                val = _clean(rec.get(opt, ""))
                if val and val not in ("nan", "None"):
                    options.append(f"{opt}. {val}")
            answer = _clean(rec.get("answer", ""))
            rows.append({
                "source": "C-Eval",
                "subject": subject_cn,
                "qid": f"ceval_{sub_dir}_{split}_{rec.get('id', len(rows))}",
                "question": question,
                "options": options,
                "answer": answer,
                "meta": {"split": split, "raw_subject": sub_dir},
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"subject": subject_cn, "stem": out_stem, "count": len(rows), "path": str(out_path)}


def _ingest_import_dir(base: Path) -> int:
    """把 data/huaci/import/ 下的手工导入题目并入分科目录。

    格式（JSONL 或 JSON 数组，一题一条）：{source, subject, question, options, answer}
    subject 取值：数学 / 语文（库帕思 K12 等数据源下载后丢进 import/ 即可）。
    返回新增题数。
    """
    imp = base / "import"
    if not imp.is_dir():
        return 0
    added = 0
    for p in sorted(imp.glob("*.json*")):
        raw = p.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except Exception as exc:
            print("  [跳过] 无法解析 %s：%s" % (p.name, str(exc)[:80]))
            continue
        if isinstance(data, dict):
            data = data.get("questions", data.get("data", []))
        for i, rec in enumerate(data or []):
            if not isinstance(rec, dict):
                continue
            subject_cn = str(rec.get("subject", "")).strip()
            if subject_cn not in ("数学", "语文"):
                continue
            question = _clean(rec.get("question", ""))
            if not question:
                continue
            options = [str(o).strip() for o in (rec.get("options") or []) if str(o).strip()]
            answer = _clean(rec.get("answer", ""))
            out_dir = base / subject_cn
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = "import_%s" % p.stem
            with open(out_dir / f"{stem}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "source": rec.get("source", "import"),
                    "subject": subject_cn,
                    "qid": f"imp_{p.stem}_{i}",
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "meta": {"raw_file": p.name},
                }, ensure_ascii=False) + "\n")
            added += 1
    return added


def main() -> int:
    base = Path(root_path("data/huaci"))
    summary = []
    for sub_dir, subject_cn, stem in SUBJECTS:
        print("抓取 %s（%s）..." % (sub_dir, subject_cn))
        try:
            info = _download_and_dump(sub_dir, base / subject_cn, stem, subject_cn)
            summary.append(info)
            print("  -> %d 题 -> %s" % (info["count"], info["path"]))
        except Exception as exc:
            print("  [错误] %s：%s" % (sub_dir, str(exc)[:200]))

    imported = _ingest_import_dir(base)
    if imported:
        print("\n导入 data/huaci/import/：新增 %d 题" % imported)

    print("\n===== 管道汇总 =====")
    for s in summary:
        print("  %s/%s: %d 题" % (s["subject"], s["stem"], s["count"]))
    if imported:
        print("  手工导入: %d 题" % imported)
    print("输出根目录:", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
