# -*- coding: utf-8 -*-
"""划词训练候选生成：给定一道题（题干+选项），生成"可划词"候选及特征。

与文章关键词不同，题目文本短（几十到几百字），TextRank/PMI 频次信号弱，
因此划词候选基于"读题时学习者会划什么"来生成，按功能分为三类：
  - data    数据类：数字、公式、单位、已知条件、具体量（学习者优先划取）
  - logic   逻辑类：设问/条件逻辑词（求/若/则/已知/下列/正确/错误…）
  - concept 概念类：学科概念/名词（函数、加速度、意象…）

训练目标（下一阶段）：
  1. 预测每题哪些候选"应划"（label 1）vs"不划"（label 0）；
  2. 划词序列按 data/logic 占比 → 读题策略（数据优先/逻辑优先/跳跃型）。

供 tools/huaci_gen_pool.py 与后续训练脚本复用。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

# 复用文章关键词的停用词（过滤虚词/语气词/标点）
from modules.semantic.keywords import _STOPWORDS

# ── 语文学科专属词典（读题会圈的核心考点/功能词） ─────────────────────
# 考点/手法/表达方式词：桩点必划（concept 高优先，博士读到立即高亮）
_YUWEN_KAODIAN = {
    # 修辞/表现手法
    "比喻", "拟人", "夸张", "排比", "对比", "衬托", "反衬", "借代", "设问", "反问",
    "反复", "对偶", "象征", "渲染", "烘托", "用典", "白描", "互文", "顶真",
    "情景交融", "直抒胸臆", "借景抒情", "托物言志", "动静结合", "虚实结合",
    "首尾呼应", "卒章显志", "铺垫", "伏笔", "照应", "悬念", "联想", "想象",
    # 表达方式
    "记叙", "描写", "抒情", "议论", "说明", "叙述",
    # 文体/结构/主旨/人物等考察对象
    "意象", "意境", "主旨", "主题", "标题", "中心", "论点", "论据", "论证",
    "形象", "性格", "情感", "语言", "结构", "细节", "线索", "顺序", "人物",
    "作用", "含义", "解说", "表达", "内涵",
}
# 语文反义/对比链（替换数学向的"正数/负数/大于/小于"）：出现即必划
_ANTONYM_CH = {
    "正面", "侧面", "直接", "间接", "明", "暗", "实", "虚", "详", "略",
    "主要", "次要", "正确", "错误", "符合", "不符合", "相符", "不相符",
    "属于", "不属于", "包含", "不包含", "静态", "动态", "动", "静",
    "褒", "贬", "今", "昔", "此", "彼", "有形", "无形",
}
# 关联词：语病定位（因为…所以、不但…而且…）
_GUANLIANCI = {
    "因为", "所以", "不但", "而且", "虽然", "尽管", "但是", "然", "却", "然而",
    "不仅", "还", "并且", "既", "只要", "就", "只有", "才", "如果", "那么",
    "即使", "也", "无论", "不管", "或者", "要么", "与其", "不如",
}
# 双面词：常与单面词误配 → 一面对两面逻辑病
_SHUANGMIAN = {
    "能否", "是否", "成败", "优劣", "好坏", "高低", "强弱", "多少", "有无",
    "能不能", "有没有", "是不是", "可不可以", "可否",
}
# 否定/多重否定：数否定符（避免/防止 + 不 → 表意相反）
_FOUDING = {
    "没有", "避免", "防止", "否认", "禁止", "杜绝", "严禁", "不得", "不再",
    "未曾", "并非", "未尝", "莫", "岂", "难道",
}
# 限制/绝对化/范围词：判断题命门（全部/都/唯一/一定/首/末/之一/部分…）
_XIANZHI = {
    "全部", "都", "唯一", "一定", "必然", "彻底", "完全", "首", "末", "之一",
    "首要", "根本", "凡是", "极为", "尤其", "绝对", "普遍", "整体", "部分",
    "个别", "某些", "往往", "一般", "处处",
}
# 介词滥用（易致缺主语）＋使令/处置动词：语病判断题的常见定位点
_YUWEN_PREP = {
    "通过", "经过", "随着", "关于", "对于", "根据", "为了", "由于", "被",
    "朝", "向", "往", "凭借",
}
_YUWEN_BA = {
    "使", "令", "让", "致使", "导致", "予以", "加以", "进行", "令人", "让人",
}
# 语病定位必划词 = 关联 / 双面 / 否定 / 介词 / 使令
_YUWEN_BING = _GUANLIANCI | _SHUANGMIAN | _FOUDING | _YUWEN_PREP | _YUWEN_BA
# 语病/判断高频定位词（排序时须压过选项里的普通名词，否则被挤出 TopK）
_YUWEN_SURFACE = _ANTONYM_CH | _XIANZHI | _YUWEN_BING
# 文言单字虚词（高频被考查对象，如"'之'字用法"）：需要单字候选保护
_WENYAN_CHAR = {
    "之", "其", "而", "则", "以", "乃", "于", "为", "与", "且", "若", "所",
    "者", "也", "焉", "乎", "因", "何", "莫", "盖", "而", "及", "夫", "故",
}
_WY_QUOTED = re.compile(r"[‘’'\"“”「」『』]([^‘’'\"“”「」『』])[‘’'\"“”「」『』]")


def _wenyany_target(question: str):
    """从题干定位"被考查的文言单字虚词"（如『之』字 / “其”字 / '之'），无则 None。

    只识别被引号/书名号包住的单字虚词，或单字虚词紧邻"字"，
    避免把普通文言里出现的虚词误判为考点。
    """
    if not question:
        return None
    for m in _WY_QUOTED.finditer(question):
        ch = m.group(1)
        if ch in _WENYAN_CHAR:
            return ch
    for ch in _WENYAN_CHAR:
        if (ch + "字") in question:
            return ch
    return None

_PUNCT = re.compile(r"[\s\W_]+")
_NUM = re.compile(r"[0-9０-９．.+-/%%·πeE]+")
_LATEX = re.compile(r"\$[^$]+\$")
# 多字单位：子串匹配可靠（不会误伤"积分/角度"等词）
_UNITS_MULTI = {
    "厘米", "毫米", "千米", "分米", "微米", "纳米", "弧度", "小时", "千克", "公斤", "毫克",
    "开尔文", "立方米", "平方米", "立方厘米", "平方厘米", "毫升", "摩尔", "兆帕", "千帕",
}
# 单字单位：仅当紧邻数字/字母（如 30度 / 2kg / m/s2）时才视为单位
_UNITS_SINGLE = {
    "米", "度", "秒", "分", "克", "吨", "牛", "帕", "瓦", "焦", "伏", "安", "欧", "库",
    "赫", "升", "角", "亩",
}
# 数字+单位组合：整体抽取为数据类候选（修复 "2kg" / "10m/s2" 被切碎）
_NUMUNIT = re.compile(
    r"(?:\d+(?:\.\d+)?|\.\d+)\s*"
    r"(?:[a-zA-Z°%‰][a-zA-Z0-9/·_.]*|[毫微纳千分厘]?[米秒克牛瓦焦伏安欧赫度升角])"
)
# 拉丁字母组合（如 SVO / SOV / DNA），整体抽取为概念候选（修复 "SVO型" 被切碎）
_LATIN = re.compile(r"[A-Za-z]{2,}")
# ASCII 单位缩写：判定"数字+单位"组合的尾部是否为真实单位（避免 "2x" 被误标带单位）
# 单字母只保留 m/s/g（数字+单字母几乎总是单位，如 2m=2米）；
# a/t/v/k/n/w/j/l/h/d 等易与变量混淆（2a=2倍a、2n=2n），不作为单位
_UNITS_ASCII = {
    # 质量
    "kg", "mg", "g",
    # 长度
    "m", "cm", "mm", "km", "dm",
    # 时间
    "s", "ms", "min",
    # 力学/电学/热学/化学
    "pa", "kpa", "mpa", "mol", "ml", "l", "hz", "rad",
    # 复合单位分解后允许出现的分量（m/s → m,s；m/s2 → m,s2→s）
}
_ASCII_TAIL = re.compile(r"[a-zA-Z°%‰][a-zA-Z0-9/·_.²³]*$")
# 数学/物理常见"已知条件/具体量"词（数据类强提示）
_DATA_HINT = {
    "半径", "直径", "面积", "体积", "周长", "长度", "宽度", "高度", "深度", "距离", "质量", "速度",
    "加速度", "时间", "温度", "角度", "比例", "概率", "频率", "浓度", "电量", "电压", "电流", "功率",
    "效率", "数量", "个数", "次数", "平均值", "方差", "样本", "数据", "数值", "值", "单位", "坐标",
    "方程", "函数", "数列", "集合", "不等式", "图形", "直线", "曲线", "圆", "三角形", "正方形",
    "矩形", "平行四边形", "梯形", "球", "圆柱", "圆锥", "棱柱", "几何体", "根号", "次数", "系数",
    "因数", "倍数", "余数", "商", "和", "差", "积", "斜率", "截距", "极值", "导数", "积分", "极限",
    "向量", "矩阵", "行列式", "特征值", "概率", "期望", "方差", "标准差", "正弦", "余弦", "正切",
    "对数", "指数", "常数", "变量", "参数", "条件", "结果", "结论", "答案", "选项",
}
# 逻辑/设问类词（逻辑优先型划词对象）
_LOGIC_WORDS = {
    "求", "若", "则", "已知", "设", "当", "且", "因为", "所以", "证明", "判断", "下列", "正确",
    "错误", "属于", "不属于", "其中", "分别", "依次", "化简", "计算", "求证", "比较", "大小",
    "关系", "满足", "使", "令", "存在", "唯一", "均", "都", "恰", "至多", "至少", "恒", "成立",
    "试", "问", "求值", "分析", "概括", "表达", "作用", "手法", "情感", "主旨", "赏析", "理解",
    "结合", "说明", "指出", "简述", "根据", "依据", "含", "最小", "最大", "最多", "最少", "首次",
    "同时", "分别", "各自", "可能", "一定", "必须", "是否", "能否", "如何", "为什么",
}


def _clean_token(tok: str) -> str:
    return tok.strip()


def _token_type(token: str, subject: str) -> str:
    """把一个词判为 data / logic / concept 三类。"""
    if _NUM.fullmatch(token) or "$" in token:
        return "data"
    if token in _DATA_HINT:
        return "data"
    if token in _LOGIC_WORDS:
        return "logic"
    return "concept"


def _ends_with_unit(tok: str) -> int:
    """判定"数字+单位"组合（如 2kg / 10m/s2 / 30度 / 2x）结尾是否确为单位。

    中文单位（30度 / 2千克）按结尾子串匹配即可；
    ASCII 单位（2kg / 10m/s2）按尾部字母部分查 _UNITS_ASCII 白名单，
    复合单位（m/s2、km/h）拆 '/' 后逐段校验；"2x" 的 x 不在名单 → 0。
    """
    for u in _UNITS_MULTI:
        if tok.endswith(u):
            return 1
    for u in _UNITS_SINGLE:
        if tok.endswith(u):
            return 1
    m = _ASCII_TAIL.search(tok)
    if m:
        tail = m.group(0).lower().replace("²", "2").replace("³", "3")
        parts = tail.split("/")
        if parts and all(_ascii_unit_ok(p) for p in parts):
            return 1
    return 0


def _ascii_unit_ok(part: str) -> bool:
    """校验复合单位的一个分量：去末尾指数/数字后应在 ASCII 单位名单内（m/s2 → s2 → s）。"""
    part = re.sub(r"[0-9]+$", "", part.lower())
    return part in _UNITS_ASCII


def _has_unit(tok: str) -> int:
    """判断词是否含单位：多字单位子串命中即可；单字单位须紧邻数字/ASCII字母。

    注意：中文汉字的 str.isalpha() 也为 True，若直接用 isalpha 判断会把
    "积分→分"、"角度→度"、"加速度→度" 误判为带单位；故只用数字与 ASCII
    字母（拉丁）作为"紧邻"证据（如 30度 / 2kg / m/s2 里的 0 / k / s）。
    """
    if any(u in tok for u in _UNITS_MULTI):
        return 1
    for u in _UNITS_SINGLE:
        idx = tok.find(u)
        while idx != -1:
            if idx > 0:
                prev = tok[idx - 1]
                if prev.isdigit() or (prev.isascii() and prev.isalpha()):
                    return 1
            idx = tok.find(u, idx + 1)
    return 0


# 设问/题目细节词（读题必看的"要求"）——用户口语称"题目细节"，如 求/最小值/比较/计算
_ASK_WORDS = {
    "求", "求值", "求解", "计算", "化简", "比较", "大小", "关系", "最大", "最小", "最大值", "最小值",
    "最多", "最少", "取值", "值域", "判断", "证明", "求证", "说明", "指出", "意义", "表示",
    "列出", "是否", "能否", "依次", "分别", "解析式", "表达式",
}
# 出现 ≥2 种不同单位时（如 km 与 m、kg 与 g 混用）视为"单位可能混淆"，单位相关候选可划
def _distinct_units(text: str) -> set:
    units = set()
    for m in _NUMUNIT.finditer(text):
        u = re.sub(r"^\d+(?:\.\d+)?\s*", "", m.group(0))
        if u:
            units.add(u)
    return units


def generate_question_candidates(
    question: str,
    options: List[str] | None = None,
    subject: str = "数学",
    top_k: int = 12,
) -> List[dict]:
    """为一道题生成划词候选。

    Args:
        question: 题干。
        options: 选项列表（可选，用于候选覆盖率与"是否出现在选项"）。
        subject: 学科（数学/语文…），影响分类提示。
        top_k: 返回候选数量上限。

    Returns:
        候选列表 [{kw, type, length, freq, first_pos_norm, in_options,
                   is_formula, is_number, has_unit, ...}]，按可划性降序。
    """
    import jieba

    opt_text = " ".join(options) if options else ""
    # 候选从题干+选项提取（语文/数学知识题的关键术语常在选项里），
    # 选项词用 in_options 标记；裸数字等噪音仍过滤。
    text = question + (" " + opt_text if opt_text else "")
    candidates: Dict[str, dict] = {}

    # 待屏蔽区间：公式 / 数字+单位 整体已是候选，屏蔽后避免 jieba 重复切出碎片
    mask = [False] * len(text)

    def _mark(lo: int, hi: int) -> None:
        for i in range(lo, hi):
            if i < len(mask):
                mask[i] = True

    def _add(kw: str, ttype: str, *, is_formula=0, is_number=0, has_unit=0,
             is_kaodian=0, span=None) -> None:
        kw = kw.strip()
        if not kw:
            return
        c = candidates.setdefault(kw, {
            "kw": kw, "type": ttype, "length": len(kw), "freq": 0,
            "first_pos_norm": 1.0, "in_options": 0,
            "is_formula": 0, "is_number": 0, "has_unit": 0, "is_kaodian": 0,
        })
        c["freq"] += 1
        c["is_formula"] = max(c["is_formula"], is_formula)
        c["is_number"] = max(c["is_number"], is_number)
        c["has_unit"] = max(c["has_unit"], has_unit)
        c["is_kaodian"] = max(c["is_kaodian"], is_kaodian)
        if span and c["first_pos_norm"] == 1.0 and len(text):
            c["first_pos_norm"] = round(span[0] / len(text), 4)

    # 1) LaTeX 公式整体作为数据类候选（题干中的公式，如 $y=\frac{k}{x}$）
    for m in _LATEX.finditer(text):
        _add(m.group(0), "data", is_formula=1, span=(m.start(), m.end()))
        _mark(m.start(), m.end())
    # 2) 数字+单位组合整体抽取为数据类候选（修复 "2kg" / "10m/s2" 被切碎）
    for m in _NUMUNIT.finditer(text):
        tok = m.group(0).replace(" ", "")  # "3 秒"→"3秒"；空格不属单位内容
        if tok and not any(u in tok for u in _UNITS_SINGLE if u == tok):  # 排除纯单字单位
            _add(tok, "data", is_number=1, has_unit=_ends_with_unit(tok),
                 span=(m.start(), m.end()))
            _mark(m.start(), m.end())
    # 3) 拉丁字母组合（SVO / SOV / DNA），整体抽取为概念候选（修复 "SVO型" 被切碎）
    for m in _LATIN.finditer(text):
        if any(mask[m.start():m.end()]):
            continue  # 已在数字+单位/公式区间内（如 m/s2）
        tok = m.group(0)
        _add(tok, "concept", span=(m.start(), m.end()))
        _mark(m.start(), m.end())  # 屏蔽后 jieba 不再从 "SVO型" 切出 "SV"/"O型" 碎片

    # 4) jieba 分词候选（屏蔽公式/数字单位区间后，避免产生 pi、2kg 等碎片）
    clean_text = "".join(" " if masked else ch for ch, masked in zip(text, mask))
    tokens = [t for t in jieba.cut(clean_text) if t.strip()]
    n_tok = max(1, len(tokens))
    pos = 0
    for tok in tokens:
        t = _clean_token(tok)
        if not t or len(t) > 8:
            pos += 1
            continue
        if _PUNCT.fullmatch(t) or t in _STOPWORDS:
            pos += 1
            continue
        # 单字裸数字（1/2/3 等）是坐标/序号噪音，不作为候选
        if _NUM.fullmatch(t) and len(t) == 1:
            pos += 1
            continue
        ttype = _token_type(t, subject)
        if ttype == "concept" and len(t) < 2:
            pos += 1
            continue  # 单字概念词噪音大，默认不作为候选
        c = candidates.setdefault(t, {
            "kw": t, "type": ttype, "length": len(t), "freq": 0,
            "first_pos_norm": 1.0, "in_options": 0,
            "is_formula": 0, "is_number": int(_NUM.fullmatch(t) is not None),
            "has_unit": _has_unit(t),
            "is_kaodian": 1 if subject == "语文" and t in _YUWEN_KAODIAN else 0,
        })
        c["freq"] += 1
        if c["first_pos_norm"] == 1.0:
            c["first_pos_norm"] = round(pos / n_tok, 4)
        pos += 1

    # 5) 语文学科：仓库词表整体抽取（jieba 可能把"情景交融/托物言志/一面对两面"等
    #    考点词拆散成碎片，这里按整词回补；功能词归为 logic（语病/判断定位词）。
    if subject == "语文":
        tokset = set(tokens)  # 单字功能词须为独立分词，避免"使用"被误抽出"使"
        _YUWEN_VOCAB = (
            [(kw, "concept", 1) for kw in _YUWEN_KAODIAN]
            + [(kw, "logic", 0) for kw in (_ANTONYM_CH | _XIANZHI | _YUWEN_BING)]
        )
        for kw, ttype, is_kaodian in _YUWEN_VOCAB:
            if len(kw) == 1 and kw not in tokset:
                continue  # "使/被/让…"须独立成词才计语病定位词
            pos_in = text.find(kw)
            if pos_in == -1:
                continue
            if any(mask[pos_in:pos_in + len(kw)]):
                continue  # 已整体成为候选（如公式/数字单位）
            _add(kw, ttype, is_kaodian=is_kaodian, span=(pos_in, pos_in + len(kw)))
            _mark(pos_in, pos_in + len(kw))
        # 文言单字虚词保护：题干"被考查"的 X 字（如 '之'字）→ 单独抽为必划候选，
        # 绕开"单字概念默认过滤"，让文言虚词题能划出考点字。
        wy = _wenyany_target(question)
        if wy:
            pos_in = text.find(wy)
            if pos_in != -1 and not any(mask[pos_in:pos_in + 1]):
                _add(wy, "concept", is_kaodian=1, span=(pos_in, pos_in + 1))

    # 6) 选项覆盖标记（候选是否也出现在选项中）
    for c in candidates.values():
        if opt_text and c["kw"] in opt_text:
            c["in_options"] = 1

    # 可划性排序（读题视角）：设问/题目细节词 > 考点词 > 概念 > 逻辑 > 数据（数字/字母不再重点）
    # 单位混淆场景（≥2 种单位混用）下，带单位的候选取"可划"级，提醒学生注意换算。
    multi_unit = len(_distinct_units(text)) >= 2

    def _score(c: dict) -> float:
        kw = c.get("kw", "")
        base = 0.0
        if kw in _ASK_WORDS:
            base += 3.6  # 设问词/题目细节（求/最小值/比较…）最高：读题必看
        if c["is_kaodian"]:
            base += 3.0  # 语文学科考点/手法/表达词
        if subject == "语文" and kw in _YUWEN_SURFACE:
            base += 1.6  # 语病/判断定位词压过普通名词，防被挤出 TopK
        if c["type"] == "concept":
            base += 2.0
        elif c["type"] == "logic":
            base += 1.2
        elif c["type"] == "data":
            base += 0.4  # 数字/字母/公式：不再作为划词重点
            if kw in _DATA_HINT and not c.get("is_number") and not c.get("is_formula"):
                base += 1.6  # "半径/体积/方程/函数"等条件属性词是读题对象，高于具体数字字母
        if c["is_formula"]:
            base += 0.3  # 公式整体略高于裸数字，但远低于概念/设问
        if c["has_unit"]:
            base += 1.4 if multi_unit else -0.4  # 多单位易混淆时才值得划单位
        if c["is_number"] and not c["has_unit"]:
            base -= 0.5  # 裸数字继续压低
        if c["in_options"]:
            base += 0.2
        return base + min(1.0, c["freq"] * 0.5) - c["first_pos_norm"] * 0.4

    out = sorted(candidates.values(), key=_score, reverse=True)
    for rank, c in enumerate(out[:top_k], start=1):
        c["rank"] = rank
    return out[:top_k]


# ── 划词预测模型：训练/推理统一的特征口径 ─────────────────────────
# 供 tools/huaci_train.py 训练、后端预测复用。特征全部来自候选行本身，
# 由题干/选项即时可算，不含 label（与关键词重排同一原则）。
_HUACI_TYPE_ID = {"data": 0, "logic": 1, "concept": 2}


def huaci_feature_names() -> list:
    """划词预测模型的特征顺序（训练与推理必须一致）。"""
    return ["t_data", "t_logic", "t_concept", "length", "freq_log",
            "first_pos_norm", "in_options", "is_formula", "is_number", "has_unit",
            "is_kaodian"]


def huaci_to_features(c) -> list:
    """把单个划词候选转成 11 维特征向量（候选生成后由模型打分）。"""
    t = _HUACI_TYPE_ID.get(str(c.get("type", "concept")), 2)
    return [
        1 if t == 0 else 0,
        1 if t == 1 else 0,
        1 if t == 2 else 0,
        float(c.get("length", len(str(c.get("kw", ""))))),
        math.log1p(float(c.get("freq", c.get("frequency", 1)) or 1)),
        float(c.get("first_pos_norm", 1.0)),
        1 if c.get("in_options") else 0,
        1 if c.get("is_formula") else 0,
        1 if c.get("is_number") else 0,
        1 if c.get("has_unit") else 0,
        1 if c.get("is_kaodian") else 0,
    ]
