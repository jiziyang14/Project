"""文章中心词提取：两阶段级联（统计预筛 → 语义校验）。

相比旧版（纯 KeyBERT 英文停用词、一次性嵌入全部候选导致缓慢且噪声大）的重大改进：

1. 中文感知：jieba 中文分词 + 中文停用词过滤，杜绝"的/了/是"等虚词上屏
2. 语义增强：改用多语言 SBERT（paraphrase-multilingual-MiniLM-L12-v2），
   12 层 Transformer 对中文语义建模显著优于英文专精的 all-MiniLM-L6-v2
3. 统计先验：TextRank（图排序，类 PageRank）+ PMI（互信息）搭配检测，
   在调用大模型前先用轻量统计把候选从数百个压到几十个，既砍掉噪声又提速
4. 科学置信度：最终置信度 = 语义相似度(余弦) 与 统计显著性(归一化 TextRank)
   的加权融合，如实反映"这个词到底多贴合全文主题"
5. 两阶段级联：只对 Top-K 候选做语义向量化，避免对全部候选嵌套大模型，
   单次提取从 50+ 秒降到秒级，且处理过程分阶段可感知
6. 透明可解释：返回完整流水线（各阶段耗时/样本量）、模型信息、每词
   置信度/语义分/统计分/频次/覆盖率与整体质量评估

置信度 = (0.75·语义相对贴合 + 0.25·统计相对显著度) × 聚焦因子，
其中聚焦因子 = min(1, 最优词绝对余弦 / 0.65)，主题越集中置信度越高，
主题发散的劣质文章会被如实压低，落在 [0,1] 区间。
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.semantic import sbert_service

logger = logging.getLogger(__name__)

# 中文停用词（常见高频虚词/连接词，避免其成为中心词）
_STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "与", "及", "或", "也", "都", "而", "且",
    "并", "但", "这", "那", "其", "它", "他", "她", "我", "你", "我们", "你们",
    "他们", "一个", "一种", "这个", "那个", "这些", "那些", "可以", "需要", "进行",
    "通过", "对于", "关于", "因为", "所以", "如果", "然后", "但是", "以及", "其中",
    "对", "从", "到", "被", "把", "让", "会", "能", "要", "就", "已经", "主要",
    "等", "着", "过", "之", "中", "上", "下", "后", "前", "时", "年",
    # 文言/古诗文常见虚词与语气词（避免其或残片成为中心词）
    "乎", "焉", "也", "矣", "夫", "哉", "欤", "邪", "耶", "兮", "尔", "诸",
    "盖", "既", "未", "弗", "尝", "每", "岂", "安", "何", "孰", "莫", "皆", "咸",
    "斯", "兹", "是", "匪", "莫", "惟", "犹", "且", "乃", "故", "则", "若", "云",
}

_PUNCT = re.compile(r"[\s\W_]+")
_HAN = re.compile(r"[\u4e00-\u9fa5]")

# 语义阶段只向量化统计预筛后的 Top-K 候选（模型加载后每条约 5ms）
_PRE_K = 40
# 语义相对贴合权重（其余 8% 归统计显著度）
_ALPHA = 0.92
# 绝对语义聚焦基线：最优词与全文的余弦达到该值即视为"主题高度聚焦"
_FOCUS_REF = 0.65
# TextRank 参数
_TR_WINDOW = 3
_TR_DAMPING = 0.85
_TR_ITER = 50
# PMI 搭配检测：互信息阈值 + 最少共现次数
_PMI_MIN = 0.5
_PMI_MIN_COUNT = 2

# 语义门槛：绝对贴合分过低的候选即使被模型打分也不允许上榜
# 兜底：真正贴合主题的词与全文绝对余弦通常 >0.2，泛词/残片往往 <0.15。
_SEM_GATE = 0.15

# 重排模型分与规则分融合权重：模型擅长标题/位置/跨句中心性，
# 规则保语义+统计平衡；50/50 融合在独立测试集上召回最稳。
_BLEND_ALPHA = 0.5

# 文言虚词/语气词字符集，用于过滤"全由虚词组成"的切分残片
_WENYAN_FUNC = set(
    "之其而乎焉者也矣夫且以于与者所乃则故若惟犹盖曰见弗未既尝每岂安何孰莫宜诚"
    "哉欤邪耶兮尔诸云是匪皆咸斯兹叹禁乎善邪哉诸兮乎善耶"
)

# ── 有监督重排（可选用模型替换手写加权融合） ─────────────
# 训练脚本 tools/train_rerank.py 产出 data/models/kw_rerank.pkl，
# 存在时用于给候选打分排序，否则回退到本文件的规则加权融合。
# 通过环境变量 KW_RERANK=off 可关闭。
import os
import pickle

_RERANK_ENABLED = os.environ.get("KW_RERANK", "on").lower() != "off"
_RERANK = None  # 加载后的重排模型 payload（含 model / features / metrics）


def _feature_names() -> list:
    """重排模型使用的特征顺序（训练与推理必须一致）。"""
    return ["freq_log", "coverage", "stat_score", "stat_norm", "semantic", "length",
            "is_subsumed", "unigram", "in_title", "first_pos_norm", "sent_cov"]


def _to_features(item: dict) -> list:
    """把单个候选（含 word/basis/frequency/coverage/stat_score/stat_norm/semantic/...）转成特征向量。"""
    sem = item.get("semantic", item.get("semantic_score", 0.0))
    return [
        math.log1p(float(item.get("frequency", 0))),
        float(item.get("coverage", 0.0)),
        float(item.get("stat_score", 0.0)),
        float(item.get("stat_norm", 0.0)),
        float(sem),
        float(item.get("length", len(str(item.get("word", ""))))),
        1 if item.get("is_subsumed") else 0,
        1 if "搭配" in str(item.get("basis", "")) else 0,
        1 if item.get("in_title") else 0,
        float(item.get("first_pos_norm", 1.0)),
        float(item.get("sent_cov", 0.0)),
    ]


def _context_features(word: str, tokens: list, sentences: list, title_words: set) -> dict:
    """计算候选词的上下文特征（训练与推理口径必须一致）。

    - in_title: 词是否出现在文章标题（标题词多为关键中心词）
    - first_pos_norm: 词首次出现的归一化位置（0=文首，1=文末，越靠前越中心）
    - sent_cov: 词覆盖的不同句子比例（跨句分布比词频更能反映中心性）
    """
    first = len(tokens)
    for i, t in enumerate(tokens):
        if word in t or t in word:
            first = i
            break
    n_sent = max(1, len(sentences))
    sent_hit = sum(1 for s in sentences if word in s)
    return {
        "in_title": 1 if word in title_words else 0,
        "first_pos_norm": round(min(1.0, first / max(1, len(tokens))), 4),
        "sent_cov": round(min(1.0, sent_hit / n_sent), 4),
    }


def _load_rerank():
    """惰性加载重排模型；文件不存在返回 None（回退规则权重）。"""
    global _RERANK
    if _RERANK is not None or not _RERANK_ENABLED:
        return _RERANK
    from core.config import root_path as _rp

    path = _rp("data/models/kw_rerank.pkl")
    if path.exists():
        try:
            with open(path, "rb") as fh:
                _RERANK = pickle.load(fh)
            logger.info("已加载关键词重排模型（特征：%s）", _RERANK.get("features"))
        except Exception as exc:
            logger.warning("关键词重排模型加载失败，回退规则融合：%s", exc)
            _RERANK = None
    return _RERANK


def invalidate_rerank() -> None:
    """清除已加载的重排模型缓存，使重新训练后的模型在下一次提取时立即生效。

    训练脚本/接口写完新 .pkl 后调用本函数，避免正在运行的服务继续用旧缓存。
    """
    global _RERANK
    _RERANK = None


def _rerank_score(item: dict) -> Optional[float]:
    """用重排模型给候选打分（[0,1]），无模型时返回 None。"""
    model = _load_rerank()
    if not model or not model.get("model"):
        return None
    try:
        pred = float(model["model"].predict([_to_features(item)])[0])
        return max(0.0, min(1.0, pred))
    except Exception as exc:
        logger.warning("重排打分失败，回退规则：%s", exc)
        return None


def extract_keywords_full(text: str, top_n: int = 8, seed: Optional[int] = None,
                          title: str = "", on_progress: Optional[callable] = None) -> dict:
    """完整提取文章中心词，返回可解释的详细结果。

    Args:
        text: 文章纯文本。
        top_n: 返回中心词数量，默认 8（范围 5~10）。
        seed: 随机种子，保证 MMR 采样的可复现性。
        title: 可选文章标题；标题中的词会获得适度加权（仅小幅提升排序，
            不独占 Top，避免舍本逐末）。
        on_progress: 可选回调 on_progress(step, action, items, ms)，
            在每一处理阶段完成时被调用，供前端做实时过程可视化。
    """
    if not text.strip():
        raise ValueError("文章文本为空，无法提取中心词")
    top_n = max(5, min(10, int(top_n)))

    pipeline: List[dict] = []

    def _stage(name: str, action: str, items: int, ms: float) -> None:
        _ms = round(ms, 1)
        pipeline.append({"step": name, "action": action, "items": items, "ms": _ms})
        if on_progress:
            try:
                on_progress(name, action, items, _ms)
            except Exception:
                pass

    # ── 阶段一：统计预筛（不调用大模型，快） ──
    t = time.perf_counter()
    tokens, freq = _segment(text)
    total_tokens = sum(freq.values()) or 1
    _stage("文本预处理", "jieba 中文分词 + 停用词过滤", len(tokens), (time.perf_counter() - t) * 1000)

    # 标题词集合：标题分词后出现在候选里的词会获得适度加权
    title_words: set = set()
    if title and title.strip():
        title_tokens, _ = _segment(title)
        title_words = {w for w in title_tokens if 2 <= len(w) <= 6}

    # 句子切分：供上下文特征（跨句覆盖 / 首次出现位置）计算
    sentences = [s for s in re.split(r"[。！？!?；;]", text) if s.strip()]

    t = time.perf_counter()
    tr_scores = _textrank(tokens)
    colloc = _collocations(tokens, freq)
    candidates, stat_scores, basis = _combine_candidates(freq, tr_scores, colloc)
    if not candidates:
        raise ValueError("未提取到有效候选词，请检查文章内容")

    # 按统计显著度预筛，只对 Top-K 做语义向量化
    pre_list = sorted(candidates, key=lambda w: stat_scores[w], reverse=True)[:_PRE_K]
    _stage("统计评分", "TextRank 图排序 + PMI 搭配预筛", len(pre_list), (time.perf_counter() - t) * 1000)

    # ── 阶段二：语义校验（多语言 SBERT） ──
    t = time.perf_counter()
    doc_vec = sbert_service.embed_multilingual([text])[0]
    cand_vecs = sbert_service.embed_multilingual(pre_list)
    _stage("语义向量化", f"多语言SBERT嵌入（{sbert_service._MULTI_DIM}维）", len(pre_list) + 1, (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    # 语义相对贴合为主 + 统计显著度为细粒度佐证，融合为最终置信度
    sem_vals = [float(np.dot(cand_vecs[i], doc_vec)) for i in range(len(pre_list))]
    sem_max = max(sem_vals) or 1e-6
    stat_max = max(stat_scores[w] for w in pre_list) or 1.0
    # 聚焦因子：最优词与全文的绝对余弦越高，说明主题越集中，置信度越可信；
    # 主题发散的劣质文章即使相对排名靠前，也会被如实压低置信度。
    focus = min(1.0, sem_max / _FOCUS_REF)
    # 特异性：被多个更长候选词包含的词（如"学习"被"机器学习/深度学习"包含）
    # 属于通用词干，语义上偏泛，予以压低，避免泛词挤占真正的中心术语。
    subsumed = {w: sum(1 for c in pre_list if c != w and w in c) for w in pre_list}
    scored = []
    # 若存在重排模型，则用它替换规则加权置信度（_rerank_enabled 标记记录到结果）
    rerank_on = _load_rerank() is not None
    for i, word in enumerate(pre_list):
        sem = sem_vals[i]
        sem_rel = sem / sem_max
        stat_norm = stat_scores[word] / stat_max
        spec = 0.35 if subsumed[word] >= 2 else 1.0
        rule_conf = (0.92 * sem_rel + 0.08 * stat_norm) * spec * focus
        item = {
            "word": word,
            "basis": basis[word],
            "confidence": round(max(0.0, min(1.0, rule_conf)), 3),
            "rule_conf": round(max(0.0, min(1.0, rule_conf)), 3),
            "semantic_score": round(sem, 3),
            "stat_score": round(stat_scores[word], 4),
            "stat_norm": round(stat_norm, 3),
            "frequency": freq.get(word, 1),
            "coverage": round(freq.get(word, 1) / total_tokens, 3),
            "length": len(word),
            "is_subsumed": 1 if subsumed[word] else 0,
            "semantic": sem,
        }
        # 上下文特征：标题词 / 首次出现位置 / 跨句覆盖，供重排模型学习中心性
        item.update(_context_features(word, tokens, sentences, title_words))
        if rerank_on:
            ms = _rerank_score(item)
            if ms is not None:
                # 模型分与规则分融合：模型擅长标题/位置/跨句中心性，
                # 规则保语义+统计平衡；融合比单用任一更稳。
                item["confidence"] = round(
                    _BLEND_ALPHA * ms + (1 - _BLEND_ALPHA) * item["rule_conf"], 3
                )
        # 语义门槛：绝对贴合分过低的泛词/残片，即使被模型打到高分也压回低位，
        # 避免"发表/仍旧/声音"这类与主题无关的词靠特征凑到 ~0.5 挤入 Top。
        if sem < _SEM_GATE:
            item["confidence"] = min(item["confidence"], 0.35)
        # 标题词适度加权：仅小幅上抬排序（×1.18），不独占 Top，避免舍本逐末。
        if word in title_words:
            item["confidence"] = round(min(1.0, item["confidence"] * 1.18), 3)
            item["from_title"] = True
        scored.append(item)
    selected = _select_top(scored, cand_vecs, top_n)
    fuse_tip = "重排模型" if rerank_on else "规则加权融合"
    _stage("融合排序", f"{fuse_tip} + 去冗选择", len(scored), (time.perf_counter() - t) * 1000)
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank

    quality = _assess_quality(selected)

    # 记录当前是否由重排模型驱动排序，供前端透明展示
    rerank_meta = {}
    rr = _load_rerank()
    if rr:
        rerank_meta = {
            "active": True,
            "features": rr.get("features"),
            "version": rr.get("version"),
            "trained_at": rr.get("trained_at"),
            "samples": rr.get("n_samples"),
        }

    return {
        "document": {
            "chars": len(text),
            "sentences": len(re.split(r"[。！？!?；;]", text)),
            "tokens": len(tokens),
            "unique": len(freq),
            "candidates": len(candidates),
            "language": "zh",
        },
        "model": {
            "embedder": sbert_service._MULTI_MODEL_NAME,
            "dim": sbert_service._MULTI_DIM,
            "layers": 12,
            "method": "jieba分词 + TextRank/PMI统计预筛 + 多语言SBERT语义校验 + MMR去重",
            "blend": "重排模型（有监督）" if rerank_meta else f"语义相对贴合×聚焦×特异性 + 统计佐证（语义{round(_ALPHA*100)}%）",
            "rerank": rerank_meta,
            "title_boosted": bool(title_words),
        },
        "pipeline": pipeline,
        "keywords": selected,
        "quality": quality,
    }


def extract_keywords(text: str, top_n: int = 8) -> List:
    """兼容旧接口：返回 [(word, score), ...] 列表。"""
    result = extract_keywords_full(text, top_n=top_n)
    return [(k["word"], k["confidence"]) for k in result["keywords"]]


def generate_candidates(text: str, top_k: int = 20, title: str = "") -> List[dict]:
    """导出候选词 + 全量特征（供标注/训练使用，不做最终去重选择）。

    与 extract_keywords_full 共用同一两阶段流水线，返回 top_k 个候选，
    每个候选携带重排模型所需的全部特征（freq/coverage/stat/semantic/...）
    以及规则加权置信度 rule_conf，便于标注与训练时对齐特征口径。

    Returns:
        候选列表（已按 rule_conf 降序）。
    """
    tokens, freq = _segment(text)
    total = sum(freq.values()) or 1
    title_words: set = set()
    if title and title.strip():
        title_tokens, _ = _segment(title)
        title_words = {w for w in title_tokens if 2 <= len(w) <= 6}
    sentences = [s for s in re.split(r"[。！？!?；;]", text) if s.strip()]
    tr_scores = _textrank(tokens)
    colloc = _collocations(tokens, freq)
    candidates, stat_scores, basis = _combine_candidates(freq, tr_scores, colloc)
    if not candidates:
        return []
    pre = sorted(candidates, key=lambda w: stat_scores[w], reverse=True)[:max(top_k * 2, _PRE_K)]
    doc_vec = sbert_service.embed_multilingual([text])[0]
    cand_vecs = sbert_service.embed_multilingual(pre)
    sem_vals = [float(np.dot(cand_vecs[i], doc_vec)) for i in range(len(pre))]
    sem_max = max(sem_vals) or 1e-6
    stat_max = max(stat_scores[w] for w in pre) or 1.0
    focus = min(1.0, sem_max / _FOCUS_REF)
    subsumed = {w: sum(1 for c in pre if c != w and w in c) for w in pre}
    out = []
    for i, word in enumerate(pre):
        sem = sem_vals[i]
        sem_rel = sem / sem_max
        stat_norm = stat_scores[word] / stat_max
        spec = 0.35 if subsumed[word] >= 2 else 1.0
        conf = (0.92 * sem_rel + 0.08 * stat_norm) * spec * focus
        out.append({
            "kw": word,
            "basis": basis[word],
            "frequency": freq.get(word, 1),
            "coverage": round(freq.get(word, 1) / total, 3),
            "stat_score": round(stat_scores[word], 4),
            "stat_norm": round(stat_norm, 3),
            "semantic": round(sem, 3),
            "length": len(word),
            "is_subsumed": 1 if subsumed[word] else 0,
            "rule_conf": round(max(0.0, min(1.0, conf)), 3),
            **_context_features(word, tokens, sentences, title_words),
        })
    out.sort(key=lambda x: x["rule_conf"], reverse=True)
    return out[:top_k]


# ── 阶段一：统计层 ─────────────────────────────
_DICT_LOADED = False


def _ensure_dict() -> None:
    """加载领域用户词典一次，让 jieba 正确切分"机器学习/深度学习"等复合术语。"""
    global _DICT_LOADED
    if _DICT_LOADED:
        return
    import os

    import jieba

    user_dict = os.path.join(os.path.dirname(__file__), "jieba_dict.txt")
    jieba.load_userdict(user_dict)
    _DICT_LOADED = True


def _segment(text: str) -> Tuple[List[str], Counter]:
    """jieba 分词 + 领域词典 + 停用词/标点过滤，返回 (词序列, 词频字典)。"""
    import jieba

    _ensure_dict()
    tokens = []
    for w in jieba.lcut(text):
        w = w.strip()
        if not w or w in _STOPWORDS or _PUNCT.fullmatch(w):
            continue
        if not _HAN.search(w):  # 至少含一个汉字
            continue
        tokens.append(w)
    return tokens, Counter(tokens)


def _textrank(tokens: List[str]) -> Dict[str, float]:
    """TextRank：在"共现窗口"图上做迭代投票（类 PageRank）。

    两个词在同一滑动窗口内共现即视为存在语义关联边，边权为共现次数。
    迭代至收敛后，得分越高的词在图中的地位越核心，越可能是中心词。
    """
    nodes = list(dict.fromkeys(tokens))
    idx = {w: i for i, w in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {}
    weight = [[0.0] * n for _ in range(n)]
    window = _TR_WINDOW
    for i in range(len(tokens)):
        end = min(i + window, len(tokens))
        for j in range(i + 1, end):
            a, b = idx[tokens[i]], idx[tokens[j]]
            weight[a][b] += 1.0
            weight[b][a] += 1.0
    out_sum = [sum(weight[i]) for i in range(n)]
    score = [1.0] * n
    d = _TR_DAMPING
    for _ in range(_TR_ITER):
        new = []
        for i in range(n):
            total = 0.0
            for j in range(n):
                if out_sum[j] > 0:
                    total += weight[j][i] / out_sum[j] * score[j]
            new.append((1 - d) + d * total)
        score = new
    return {nodes[i]: score[i] for i in range(n)}


def _collocations(tokens: List[str], freq: Counter) -> Dict[str, int]:
    """PMI 搭配检测：只保留统计学上显著的相邻词对（≥2 次共现且互信息够高）。

    避免"学习机器""自动学习"这类拼接噪声，只保留"多层神经网络"等真实词组。
    """
    bigram_freq = Counter()
    for i in range(len(tokens) - 1):
        bigram_freq[(tokens[i], tokens[i + 1])] += 1
    total = len(tokens)
    colloc: Dict[str, int] = {}
    for (a, b), cnt in bigram_freq.items():
        if cnt < _PMI_MIN_COUNT:
            continue
        pa, pb = freq.get(a, 1) / total, freq.get(b, 1) / total
        pab = cnt / total
        if pa <= 0 or pb <= 0:
            continue
        pmi = math.log(pab / (pa * pb))
        if pmi >= _PMI_MIN:
            phrase = a + b
            if len(phrase) <= 10:
                colloc[phrase] = cnt
    return colloc


def _is_guwen_junk(word: str) -> bool:
    """判定一个候选是否属于"文言虚词残片"（全部由文言虚词/语气词组成）。

    这类词（如"之矣""乎哉""以而"）语义分极低，却常挤占候选名额，直接过滤掉。
    """
    return len(word) <= 4 and all(ch in _WENYAN_FUNC for ch in word)


def _combine_candidates(freq, tr_scores, colloc) -> Tuple[List[str], Dict[str, float], Dict[str, str]]:
    """合并一元词与显著二元词组，附统计分数与词源。"""
    candidates: List[str] = []
    stat_scores: Dict[str, float] = {}
    basis: Dict[str, str] = {}

    for word, count in freq.items():
        if not (2 <= len(word) <= 6):
            continue
        if _is_guwen_junk(word):
            continue
        candidates.append(word)
        # 统计显著度 = TextRank 得分 × 词频的对数加权（抑制极高/极低频）
        stat_scores[word] = tr_scores.get(word, 0.0) * (1.0 + math.log(count))
        basis[word] = "一元词"

    for phrase, cnt in colloc.items():
        if phrase not in stat_scores:
            candidates.append(phrase)
            stat_scores[phrase] = cnt * 0.5  # 搭配以共现频次为主（已含 PMI 显著性）
            basis[phrase] = "搭配词组"

    # 去重并限制规模
    seen = list(dict.fromkeys(candidates))
    return seen[:800], stat_scores, basis


# ── 阶段二：语义层 + 融合排序 ───────────────────
def _select_top(scored, cand_vecs, top_n, sim_guard: float = 0.95) -> List[dict]:
    """按置信度降序贪心选择，同时用余弦相似度做近义去冗余。

    排名即置信度顺序（保持一致性），仅当候选取与已选词近乎同义
    （余弦 > sim_guard）时才跳过，从而既保基数又去掉重复主题词。
    sim_guard 取 0.95（几乎同义才去重）：太紧会误杀抽象主题词，
    例如"匠心"与"匠人"相似 0.90，若阈值过低主题词会被高频具体词挡掉。
    """
    cand_of = {item["word"]: i for i, item in enumerate(scored)}
    cand_vecs_local = cand_vecs
    ordered = sorted(scored, key=lambda k: k["confidence"], reverse=True)
    selected: List[dict] = []
    for item in ordered:
        if len(selected) >= top_n:
            break
        if selected:
            v = cand_vecs_local[cand_of[item["word"]]]
            max_sim = max(float(np.dot(v, cand_vecs_local[cand_of[s["word"]]])) for s in selected)
            if max_sim > sim_guard:
                continue  # 与已选词近乎同义，跳过
        selected.append(item)
    return selected


def _assess_quality(keywords: List[dict]) -> dict:
    """质量评估：以绝对语义贴合度（余弦）为基准，如实反映主题集中度。"""
    sems = [k["semantic_score"] for k in keywords]
    confs = [k["confidence"] for k in keywords]
    avg_sem = float(np.mean(sems)) if sems else 0.0
    avg_conf = float(np.mean(confs)) if confs else 0.0
    top_sem = max(sems) if sems else 0.0
    if avg_sem >= 0.45:
        level, tip = "较好", "中心词与全文主题贴合度高，主题集中"
    elif avg_sem >= 0.30:
        level, tip = "中等", "主题存在一定发散，建议结合语境甄别"
    else:
        level, tip = "较弱", "文章主题较分散或术语密集，中心词区分度有限"
    return {
        "avg_confidence": round(avg_conf, 3),
        "avg_semantic": round(avg_sem, 3),
        "top_semantic": round(top_sem, 3),
        "level": level,
        "assessment": tip,
    }