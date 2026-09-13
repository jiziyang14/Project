"""真实语料批量下载器（维基百科知识文 + 维基文库公版文学）。

数据均为真实人类撰写、可研究使用：
- 维基百科：科普/知识/说明文（CC BY-SA）→ data/texts/w_*.txt
- 维基文库：公版文学作品（散文/小说/文言/诗）→ data/texts/lit_*.txt

用法：
    python -m tools.fetch_corpus                # 抓取全部内置清单
    python -m tools.fetch_corpus --limit 10     # 只抓前 10 条（测试）
    python -m tools.fetch_corpus --skip-import  # 只抓取，不自动入库/清理

每次抓取自动：清洗去重->截断到合理长度->保存；抓完调用导入并把
旧 AI 生成的前缀为 a<数字>_ 的篇目从语料库与文件夹移除（保持纯真实语料）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from core.config import root_path

UA = "AILearningMate/1.0 (local keyword labeling corpus; contact none)"
TEXT_DIR = root_path("data/texts")
MAX_CHARS = 1300  # 单篇保留长度（篇目太短不适用关键句标注）

# ── 真实知识/说明文（维基百科） ───────────────
WIKI_TOPICS = [
    "人工智能", "大数据", "云计算", "物联网", "自动驾驶", "机器人", "卫星定位",
    "基因编辑", "遗传病", "蛋白质组装", "碱基对", "新陈代谢", "脑科学", "神经可塑性",
    "黑洞", "引力波", "狭义相对论", "暗物质", "行星形成", "太阳系", "潮汐", "地震",
    "火山", "温室效应", "可再生能源", "光合作用", "湿地", "物种灭绝", "食物链",
    "城市化", "人口普查", "对外贸易", "货币政策", "财政政策", "通货膨胀", "资本市场",
    "数字支付", "物流", "奥林匹克运动会", "足球", "京剧", "书法",
    "水墨画", "瓷器", "茶", "造纸术", "印刷术", "火药", "郑和下西洋",
    "京杭大运河", "科举制度", "长城", "都江堰", "青铜器", "编年史",
    # 补充：理工医药农林
    "光速", "绝对零度", "量子纠缠", "夸克", "电子", "原子核", "遗传密码", "染色体",
    "DNA", "酶", "光合作用", "细胞膜", "神经元", "血液循环", "免疫", "疫苗",
    "麻醉", "阿尔茨海默病", "病毒", "细菌", "真菌", "碳中和", "绿洲", "大气层",
    "洋流", "大陆漂移", "冰川", "珊瑚礁", "热带雨林", "荒漠化", "粮食安全", "农业",
    "水稻", "玉米", "小麦", "丝绸", "茶叶", "珍珠", "任意球", "围棋", "中国象棋",
    "二十四节气", "春节", "中秋节", "端午节", "元宵节", "故宫", "天坛", "兵马俑",
    # 补充：人文社科
    "丝绸之路", "茶马古道", "长城遗址", "徽商", "晋商", "四大发明", "指南针",
    "唐诗", "宋词", "元曲", "明清小说", "本草纲目", "西游记", "红楼梦", "三国演义",
    "水浒传", "论语", "道德经", "孙子兵法", "史记", "资治通鉴", "战国策", "诗经",
]

# ── 公版文学（维基文库，作者均已进入公版） ───
LIT_TOPICS = [
    # 古典名文 / 文言
    "桃花源记", "五柳先生传", "岳阳楼记", "醉翁亭记", "小石潭记", "马说",
    "爱莲说", "陋室铭", "记承天寺夜游", "滕王阁序", "出师表", "劝学",
    "孔雀东南飞", "将进酒", "春江花月夜", "琵琶行", "赤壁赋", "兰亭集序",
    "醉翁亭记", "阿房宫赋", "六国论", "师说", "伤仲永", "湖心亭看雪",
    # 古典补充
    "观沧海", "短歌行", "蜀道难", "梦游天姥吟留别", "木兰诗", "茅屋为秋风所破歌",
    "水调歌头·明月几时有", "念奴娇·赤壁怀古", "天净沙·秋思", "山坡羊·潼关怀古",
    "登高", "黄鹤楼", "石壕吏", "蒹葭", "关雎", "出师表",
    # 现代白话散文（朱自清，1948年卒，已公版）
    "背影", "春", "匆匆", "荷塘月色", "绿", "桨声灯影里的秦淮河", "给亡妇",
    "温州的踪迹", "扬州的夏日", "南京", "说扬州",
    # 现代白话小说/回忆散文（鲁迅，1936年卒，已公版）
    "社戏", "故乡", "孔乙己", "藤野先生", "药", "一件小事", "从百草园到三味书屋",
    "阿Q正传", "狂人日记", "祝福", "伤逝", "孤独者", "在酒楼上", "朝花夕拾", "示众",
    # 新文学（老舍1966卒、郁达夫1945卒、许地山1941卒、徐志摩1931卒、周作人1967卒）
    "故都的秋", "济南的冬天", "落花生", "匆匆的旅人", "风筝", "荷塘月色",
]


def _fetch_extract(base: str, title: str) -> str:
    """通过 MediaWiki extracts API 取纯文本正文（带重试，规避限流）。"""
    url = base + "/w/api.php?action=query&format=json&prop=extracts&explaintext=1&redirects=1&titles=" + \
        urllib.parse.quote(title)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for p in pages.values():
                return p.get("extract", "") or ""
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return ""


def _fetch_wikitext(base: str, title: str) -> str:
    """取原始 wikitext（维基文库常用；含重定向手动跟随，带重试）。"""
    url = base + "/w/api.php?action=query&format=json&prop=revisions&rvprop=content&rvslots=main&titles=" + \
        urllib.parse.quote(title)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for p in data.get("query", {}).get("pages", {}).values():
                if "revisions" in p:
                    raw = p["revisions"][0]["slots"]["main"]["*"]
                    m = re.match(r"^#\s*REDIRECT\s*\[\[([^\]|]+)", raw, re.I)
                    if m:
                        return _fetch_wikitext(base, m.group(1).strip())
                    return raw
            return ""
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return ""


def _wikitext_to_text(raw: str) -> str:
    """把 wikitext 粗略清洗为纯文本（去掉模板/表格/链接标记/引用等）。"""
    text = raw
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)  # 引用
    text = re.sub(r"<ref[^>]*/?>", "", text)
    text = re.sub(r"<poem[^>]*>|</poem>", "\n", text, flags=re.I)
    text = re.sub(r"<[a-zA-Z/][^>]*>", "", text)
    text = re.sub(r"\{\{(?!page|Pages).*?\}\}", "", text, flags=re.S)  # 模板(保留排版无关的)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)  # 表格
    text = re.sub(r"\[\[(?:File|文件|Image|图)[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)  # [[a|b]]->b
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'+", "", text)  # 粗斜体
    text = re.sub(r"^\s*[=|！].*$", "", text, flags=re.M)  # 表头残留
    return text


def _fetch_text(base: str, title: str) -> str:
    """优先 extracts(纯文本)，不足再回退维基库原始 wikitext。"""
    txt = _fetch_extract(base, title)
    if len(txt) >= 60:
        return txt
    raw = _fetch_wikitext(base, title)
    return _wikitext_to_text(raw)


def _clean(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^(={1,6}\s*.*?\s*={1,6})$", s):  # 章节标题(==xx==)
            continue
        if re.match(r"^【.{0,6}】$", s):
            continue
        lines.append(s)
    return " ".join(lines)


def _trim(text: str, limit: int = MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # 在最近的句子结束符截断
    m = re.search(r"[。！？!?；;](?=[^。！？!?；;]{0,80}$)", cut)
    if m:
        cut = cut[: m.end()]
    return cut


def _title(src: str) -> str:
    out = re.sub(r"[\\/:*?\"<>|\s]+", "", src) or "未命名"
    return out[:30]


def fetch(title: str, base: str, prefix: str) -> str | None:
    text = _fetch_text(base, title)
    if not text:
        return None
    text = _trim(_clean(text))
    if len(text) < 60:
        return None
    path = TEXT_DIR / f"{prefix}_{_title(title)}.txt"
    path.write_text(text, encoding="utf-8")
    return title


def purge_ai_generated() -> tuple[int, int]:
    """移除旧 AI 生成篇目（文件名前缀 a<数字>_）与语料库对应条目。"""
    removed_files = 0
    for p in TEXT_DIR.glob("a*_*.txt"):
        try:
            p.unlink()
            removed_files += 1
        except OSError:
            pass
    # 语料库清理
    art = root_path("data/training/articles.jsonl")
    if not art.exists():
        return removed_files, 0
    kept, dropped = [], 0
    for line in art.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if re.match(r"^a\d+_", r.get("title", "")):
            dropped += 1
            continue
        kept.append(r)
    if dropped:
        art.write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in kept) + "\n", encoding="utf-8")
    return removed_files, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只抓前 N 条（测试用）")
    ap.add_argument("--only", choices=["wiki", "lit"], default=None, help="只抓维基百科/维基文库")
    ap.add_argument("--import", dest="do_import", action="store_true", help="抓完后自动导入语料库")
    ap.add_argument("--skip-import", action="store_true", help="只抓取，不做导入与清理")
    args = ap.parse_args()

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(t, "https://zh.wikipedia.org", "w") for t in WIKI_TOPICS] + \
           [(t, "https://zh.wikisource.org", "lit") for t in LIT_TOPICS]
    if args.only == "wiki":
        todo = [x for x in todo if x[2] == "w"]
    elif args.only == "lit":
        todo = [x for x in todo if x[2] == "lit"]
    if args.limit:
        todo = todo[: args.limit]

    ok, fail = [], []
    for i, (title, base, prefix) in enumerate(todo, 1):
        try:
            got = fetch(title, base, prefix)
        except Exception as exc:  # 网络/解析异常：跳过，不中断整体
            got = None
        if got:
            ok.append(got)
            print(f"[{i}/{len(todo)}] OK  {title}")
        else:
            fail.append(title)
            print(f"[{i}/{len(todo)}] SKIP {title}")
        time.sleep(0.5)

    print(f"\n抓取完成：成功 {len(ok)}，失败/跳过 {len(fail)}")

    if args.skip_import:
        return 0

    # 清理旧 AI 生成篇目，保证纯真实语料
    rf, rd = purge_ai_generated()
    print(f"已移除 AI 生成篇目：文件 {rf}，语料库条目 {rd}")

    from backend.training import import_folder
    res = import_folder()
    print(f"导入语料库：新增 {res['imported']}，跳过 {res['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())