# -*- coding: utf-8 -*-
"""扫描 all_docs.txt（10 万篇）统计全局词频表，供重排模型补“全局知识”。

产出：
  data/models/global_wordfish.pkl —— {n_docs, doc_freq, total_freq, built_at}
  其中 doc_freq[word] = 多少篇文档含该词；total_freq[word] = 全库累计出现次数。

用途（对齐《参考 wzh0211 神策杯方案》的两个强跨文档特征）：
  - 全局文档频率 → 推导 IDF = log(N / (doc_freq+1))，衡量“稀有度”；
  - 全局累计频率/total_freq → 衡量“全站常不常当成重点词”。

切词完全复用 modules.semantic.keywords._segment（jieba + 同一停用词/领域词典），
保证统计口径与推理/训练一致，避免训练/推理漂移。

用法：py -3.12 -m tools.build_global_wordfish
"""
from __future__ import annotations

import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import root_path  # noqa: E402
from modules.semantic.keywords import _segment  # noqa: E402

ALL_DOCS_PAT = root_path("data/all_docs.txt")
OUT_PKL = Path(root_path("data/models/global_wordfish.pkl"))


def main() -> int:
    if not ALL_DOCS_PAT.exists():
        print("[Error] 找不到 %s" % ALL_DOCS_PAT)
        return 1

    t0 = time.time()
    doc_freq: Counter = Counter()      # 词 -> 出现该词的文档数
    total_freq: Counter = Counter()    # 词 -> 全库累计出现次数
    n_docs = 0

    with open(ALL_DOCS_PAT, "r", encoding="utf-8", errors="ignore") as fp:
        line = fp.readline()
        while line:
            line = line.rstrip("\n")
            if line:
                parts = line.split("\x01")
                if len(parts) != 3:
                    line = fp.readline()
                    continue
                body = parts[2]
                tokens = _segment(body)[0]
                if tokens:
                    doc_freq.update(set(tokens))
                    total_freq.update(tokens)
                    n_docs += 1
            line = fp.readline()
            if n_docs % 20000 == 0 and n_docs:
                print("  processed docs=%d  unique=%d  t=%.0fs" % (n_docs, len(doc_freq), time.time() - t0), flush=True)

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_docs": n_docs,
        "doc_freq": dict(doc_freq),
        "total_freq": dict(total_freq),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT_PKL, "wb") as fh:
        pickle.dump(payload, fh)

    print("完成：docs=%d  unique_words=%d  耗时 %.0fs" % (n_docs, len(doc_freq), time.time() - t0))
    print("已写入 %s" % OUT_PKL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())