# -*- coding: utf-8 -*-
"""划词候选离线标注（读题视角口径）。

为 data/huaci/{学科}/pool.json 的每道题候选生成 label 并落盘 labels.jsonl，
供后端标注页复核/续改，兼作划词模型的训练底稿。

读题视角（本文件核心口径）：
  · 设问/题目细节词（求/比较/大小/最小值/判断…）→ 必划(2)：读题先看"要求什么"
  · 学科概念/考点词（反比例/方程/手法/主旨…）→ 必划(2)：承载考点
  · 条件量名词（半径/体积/距离…）→ 可划(1)：读题留意
  · 具体数据（数字/字母/公式式子/单一单位数值）→ 不划(0)：不是读题要点
  · 多种单位混用（≥2 种如 km 与 m）时，带单位候选 → 可划(1)：提醒换算
  · 连接/引导词（若/则/已知/若点…）→ 不划(0)：不成读题重点
每个候选的 label 附 note 理由（教学口吻，无模型术语），供人工复核。

label 语义与后端标注页一致：0=不划(压掉) 1=可划但次要 2=必划(重点)。

用法：py -3.12 -m tools.huaci_autolabel [--all] [--limit N]
  --all     全池标注（默认即全池；--limit 仅用于局部试标）
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.training import huaci_save  # noqa: E402
from core.config import root_path  # noqa: E402
from modules.semantic import huaci_candidates as hc  # noqa: E402

SUBJECTS = ["数学", "语文"]

# 语文"高频填充词"判决阈值：选项概念在 ≥_OPT_DF 道题里出现 → 压为不划(0)
_OPT_THRESHOLD = 2
_OPT_DF: Counter = Counter()      # 由 _run 在全池上预计算：候选词 → 出现的题目数
_ENTITY_WORDS = set()             # 人名/地名/书名/专名：df 压噪时免压

# ── 语文学科词典（与候选生成器保持一致） ──
_YUWEN_KAODIAN = hc._YUWEN_KAODIAN
_ANTONYM_CH = hc._ANTONYM_CH
_XIANZHI = hc._XIANZHI
_YUWEN_BING = hc._YUWEN_BING
_WENYAN_CHAR = hc._WENYAN_CHAR
_wenyany_target = hc._wenyany_target
_ASK_WORDS = getattr(hc, "_ASK_WORDS", set())
_distinct_units = hc._distinct_units
_NUM_RE = re.compile(r"[0-9０-９．.+-/%%·πeE]+")

# ── 数学：承载考点的学科概念（而非条件量） ──
_MATH_CORE = {
    "函数", "方程", "数列", "不等式", "集合", "命题", "等式", "分式", "图象", "图形",
    "三角形", "四边形", "圆", "直角", "平行", "垂直", "对称", "相似", "全等", "概率",
    "统计", "根", "解", "定义域", "值域", "单调", "奇偶", "周期", "向量", "坐标",
}
# 数学：设问引导词（判断对象题型命门）
_MATH_GATE = {
    "下列", "以下", "正确的是", "错误的是", "正确的是", "不正确的是", "正确的", "有稳定性",
    "无意义", "有意义", "能够", "不能", "正确的是", "错误的是",
}
# 连接/引导词：读题扫过即可，不是划词对象
_CONNECTOR = {
    "若点", "若", "则", "当", "已知", "设", "使得", "其中", "那么", "且", "或", "均在",
    "分别", "依次", "如图", "分别", "下列", "以下", "根据", "现有", "考虑",
}


def _multi_unit(question: str, options: list) -> bool:
    """题干+选项中是否出现 ≥2 种不同单位（易换算混淆）。"""
    text = question + " " + " ".join(options or [])
    return len(_distinct_units(text)) >= 2


def _judge_math(c: dict, question: str = "", multi_unit: bool = False) -> tuple[int, str]:
    """数学标注（读题视角）：设问与概念重点划，数据与连接词压掉。"""
    kw = c.get("kw", "")
    L = c.get("length", len(kw))
    is_formula = c.get("is_formula", 0)
    has_unit = c.get("has_unit", 0)
    is_number = c.get("is_number", 0)
    in_options = c.get("in_options", 0)
    ttype = c.get("type", "concept")

    # 设问/题目细节词：读题必看
    if kw in _ASK_WORDS:
        return 2, "题目要求，读题必看"
    # 反义对比/判断命门
    if kw in _MATH_GATE or kw in _ANTONYM_MATH:
        return 2, "判断/对比要点，读题需圈"
    # 具体数据：数字字母式子不划
    if is_formula:
        return 0, "数字字母式子，读题不必圈"
    if is_number:
        return 0, "具体数值，读题不必圈"
    if has_unit:
        return (1, "题中单位不止一种，划下防换算错") if multi_unit else (0, "数值条件，读题不必圈")
    # 衔接词：不是要点
    if kw in _CONNECTOR:
        return 0, "连接引导词，不是读题重点"
    if ttype == "concept":
        if L == 1:
            return 0, "单字，信息不足"
        if in_options:
            return 1, "选项内容，可划作备选比对"
        return 2, "题干概念，考点所在"  # 条件量已由 _DATA_HINT 归 data，此处多为真概念
    if ttype == "data":  # 属性词（半径/体积/值…）
        if kw in _MATH_CORE:
            return 2, "学科概念，考点所在"
        return 1, "条件量名词，读题留意"
    return 0, "其他，不作划词"


_ANTONYM_MATH = {
    "正数", "负数", "正", "负", "大于", "小于", "不大于", "不小于", "大于等于", "小于等于",
    "至少", "至多", "最大", "最小", "正确", "错误", "相反数", "相反", "剩余", "不足",
    "相等", "不等", "相同", "不同", "增", "减", "有", "无", "成立", "不变",
}


def _judge_chinese(c: dict, question: str = "") -> tuple[int, str]:
    """语文标注（读题视角）：考点/手法/语病定位/判断限制词必划，数据与填充词压掉。"""
    kw = c.get("kw", "")
    L = c.get("length", len(kw))
    is_formula = c.get("is_formula", 0)
    is_number = c.get("is_number", 0)
    is_kaodian = c.get("is_kaodian", 0)
    in_options = c.get("in_options", 0)
    ttype = c.get("type", "concept")

    if is_formula or is_number:
        return 0, "数字符号，读题不必圈"
    if kw in _ASK_WORDS:
        return 2, "题目要求，读题必看"
    if is_kaodian:
        return 2, "考点术语，读题必圈"
    if kw in _ANTONYM_CH:
        return 2, "反义对比点，读题需圈"
    if kw in _YUWEN_BING:
        return 2, "语病定位词，读题必圈"
    if kw in _XIANZHI and L > 1:
        return 2, "判断限制词，读题命门"
    # 文言被考单字
    if L == 1 and kw in _WENYAN_CHAR and kw == _wenyany_target(question):
        return 2, "文言被考字，读题必圈"
    if L == 1:
        return 0, "单字，信息不足"
    if ttype == "concept" and in_options and kw not in _ENTITY_WORDS \
            and _OPT_DF.get(kw, 0) >= _OPT_THRESHOLD:
        return 0, "选项高频填充词，不必圈"
    if kw in _CONNECTOR:
        return 0, "连接引导词，不是读题重点"
    return 1, "普通名词，可划作备选"


def _run(subject: str, limit: int, pool: list) -> dict:
    target = pool[:limit] if limit > 0 else pool
    labeled_q = 0
    sample_rows = 0
    dist = {0: 0, 1: 0, 2: 0}
    for q in target:
        cands = q.get("candidates", [])
        if not cands:
            continue
        multi_unit = _multi_unit(q.get("question", ""), q.get("options") or [])
        for c in cands:
            if subject == "语文":
                lbl, note = _judge_chinese(c, q.get("question", ""))
            else:
                lbl, note = _judge_math(c, q.get("question", ""), multi_unit)
            c["label"] = lbl
            c["note"] = note
            dist[lbl] = dist[lbl] + 1
        huaci_save(subject, q["qid"], cands)
        labeled_q += 1
        sample_rows += len(cands)
    return {"subject": subject, "labeled_questions": labeled_q, "samples": sample_rows, "dist": dist}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="全池标注（默认即全池）")
    ap.add_argument("--limit", type=int, default=0, help="只标前 N 题（0=全池）")
    args = ap.parse_args()
    limit = args.limit if args.limit > 0 else -1

    for subj in SUBJECTS:
        pool_path = root_path(f"data/huaci/{subj}/pool.json")
        pool = json.loads(pool_path.read_text(encoding="utf-8"))

        # 语文预计算"选项高频填充词"（全池 df）
        df = Counter()
        for q in pool:
            seen = set()
            for c in q.get("candidates", []):
                kw = c.get("kw", "")
                if (c.get("type") == "concept" and c.get("in_options")
                        and len(kw) > 1 and kw not in _YUWEN_KAODIAN and kw not in _ANTONYM_CH):
                    seen.add(kw)
            for w in seen:
                df[w] += 1
        _OPT_DF.clear()
        _OPT_DF.update(df)

        # 实体词保护（人名/地名/机构/专名 + 拉丁缩写）
        from jieba import posseg as _pseg
        ent = set()
        for q in pool:
            for c in q.get("candidates", []):
                kw = c.get("kw", "")
                if not kw or kw.isdigit():
                    continue
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", kw) and not kw.islower():
                    ent.add(kw)
                    continue
                for _tok, _flag in _pseg.cut(kw):
                    if _flag in ("nr", "ns", "nt", "nz"):
                        ent.add(kw)
                        break
        _ENTITY_WORDS.clear()
        _ENTITY_WORDS.update(ent)

        r = _run(subj, limit, pool)
        print("%s：标注 %d 题 / %d 样本（label0=%d label1=%d label2=%d）"
              % (r["subject"], r["labeled_questions"], r["samples"],
                 r["dist"][0], r["dist"][1], r["dist"][2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())