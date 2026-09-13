"""P0 流程调度器：串联录入→语义→感知→诊断→干预→融合。

调度器持有运行期共享状态（题库、采集器、Boss 状态等），
各模型以 api 隔离接口接入，单模块异常被捕获并记录，不影响主流程。
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np

from core.logger import logger
from core.schemas import Question
from core.storage import QuestionStore

logger = logging.getLogger(__name__)

# F5-1 DQN 动作说明：0=降难度 / 1=保持 / 2=升难度
DQN_ACTION_LABELS = {0: "降难度", 1: "保持", 2: "升难度"}

# ── 答题口径：宽松判题 ──────────────────────────────
# 统一全/半角标点
_FULL_HALF = str.maketrans({
    "：": ":", "；": ";", "，": ",", "、": ",", "　": " ",
    "（": "(", "）": ")", "【": "[", "】": "]", "“": '"', "”": '"',
    "‘": "'", "’": "'", "！": "!", "？": "?", "﹒": ".",
})
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_OPT_RE = re.compile(r"[a-fA-F]")
_NUM_EPS = 1e-6


def _answer_normalize(s: str) -> str:
    """规整答案：去首尾空白、统一全角、去全部空白、小写化（兼容选项字母）。"""
    if not s:
        return ""
    s = s.strip().lower().translate(_FULL_HALF)
    return re.sub(r"\s+", "", s)


def answer_matches(user: str, correct: str) -> bool:
    """判定作答是否正确（答题口径：容忍格式/单位/顺序/括号细节差异）。

    规则优先级：精确一致 → 选项字母命中等价 → 数值等价（忽略单位与写法）。
    """
    u = _answer_normalize(user or "")
    c = _answer_normalize(correct or "")
    if not u or not c:
        return u == c
    if u == c:
        return True
    # 选项题：只要用户给出的字母集合与标准答案一致即判对（无视大小写/序号）
    ul = set(_OPT_RE.findall(u))
    cl = set(_OPT_RE.findall(c))
    if cl and ul and ul == cl:
        return True
    # 数值题：提取全部数字（含小数/负数），集合等价即判对（忽略单位/分隔/顺序）
    un = [float(x) for x in _NUM_RE.findall(u)]
    cn = [float(x) for x in _NUM_RE.findall(c)]
    if cn and un:
        if len(un) == len(cn):
            return all(abs(a - b) < _NUM_EPS for a, b in zip(sorted(un), sorted(cn)))
        if len(cn) == len(set(cn)):
            return set(un) == set(cn)
    # 纯文本答案兜底：精确比较
    return u == c


class Scheduler:
    """系统调度器，维护 P0 核心闭环的运行期状态。

    Args:
        store: 题库存储实例。
    """

    def __init__(self, store: Optional[QuestionStore] = None) -> None:
        self.store = store or QuestionStore()
        # 采集模块开关（用户主导，默认全关）
        self.modules_enabled: Dict[str, bool] = {
            "mouse": False,
            "screenshot": False,
            "reading": False,
            "ocr": False,
        }
        self.mouse_collector = None
        self.screen_collector = None
        self.reading_collector = None
        self.boss_state = {"level": 1, "hp": 100, "consecutive_correct": 0, "difficulty": 1}

    def toggle_module(self, name: str) -> bool:
        """切换采集模块开关。

        Args:
            name: 模块名（mouse/screenshot/ocr）。

        Returns:
            切换后的开关状态。

        Raises:
            KeyError: 模块名非法。
        """
        if name not in self.modules_enabled:
            raise KeyError(f"未知模块：{name}")
        self.modules_enabled[name] = not self.modules_enabled[name]
        state = self.modules_enabled[name]
        logger.info("模块 %s 已%s", name, "开启" if state else "关闭")
        if name == "mouse":
            self._sync_mouse(state)
        elif name == "screenshot":
            self._sync_screen(state)
        elif name == "reading":
            self._sync_reading(state)
        return state

    def _sync_reading(self, enabled: bool) -> None:
        """同步划词采集器生命周期（事件式，无后台线程）。"""
        if enabled:
            if self.reading_collector is None:
                from modules.collect.reading import ReadingCollector

                self.reading_collector = ReadingCollector()
            self.reading_collector.start()
        elif self.reading_collector is not None:
            self.reading_collector.stop()

    def _sync_mouse(self, enabled: bool) -> None:
        """同步鼠标采集器生命周期。"""
        if enabled:
            if self.mouse_collector is None:
                from modules.collect.mouse import MouseCollector

                self.mouse_collector = MouseCollector()
            self.mouse_collector.start()
        elif self.mouse_collector is not None:
            self.mouse_collector.stop()

    def _sync_screen(self, enabled: bool) -> None:
        """同步截图采集器生命周期。"""
        if enabled:
            if self.screen_collector is None:
                from modules.collect.screen import ScreenCollector

                self.screen_collector = ScreenCollector()
            self.screen_collector.start()
        elif self.screen_collector is not None:
            self.screen_collector.stop()

    def add_question(self, **kwargs) -> Question:
        """新增一道题并触发聚类增量更新。

        Args:
            **kwargs: 传入 QuestionStore.add_question 的参数。

        Returns:
            新建的题目。
        """
        question = self.store.add_question(**kwargs)
        # 触发题型聚类（题目数足够时），异常不影响主流程
        try:
            if self.store.count() >= 5:
                self._cluster_types()
        except Exception as exc:
            logger.error("题型聚类更新失败：%s", exc)
        return question

    def submit_answer(self, question_id: str, user_answer: str, confidence_self: int) -> dict:
        """处理答题提交并触发错题分析链。

        Args:
            question_id: 题目标识。
            user_answer: 用户作答。
            confidence_self: 自评 0/1。

        Returns:
            判定与触发分析的结果摘要。
        """
        question = self.store.get(question_id)
        if question is None:
            raise KeyError(f"题目不存在：{question_id}")
        is_correct = answer_matches(user_answer, question.correct_answer)
        question.user_answer = user_answer
        question.confidence_self = confidence_self
        question.is_correct = is_correct
        question.is_wrong = not is_correct
        question.is_mastered = is_correct
        self.store._flush()

        result = {"is_correct": is_correct, "correct_answer": question.correct_answer}
        if is_correct:
            self.boss_state["consecutive_correct"] += 1
            decision = self._dqn_decision(heuristic_delta=1)
            self.boss_state["difficulty"] = min(5, max(1, self.boss_state["difficulty"] + decision["delta"]))
            result["boss"] = self._boss_payload(decision)
            return result

        # 答错 → 错题分析链
        self.boss_state["consecutive_correct"] = 0
        decision = self._dqn_decision(heuristic_delta=-1)
        self.boss_state["difficulty"] = min(5, max(1, self.boss_state["difficulty"] + decision["delta"]))
        result["boss"] = self._boss_payload(decision)
        result["analyses"] = self._run_wrong_analyses(question)
        return result

    def _dqn_is_stuck(self) -> float:
        """根据鼠标采集器判断当前是否"卡壳"（长停顿或位移极小）。

        Returns:
            0.0 表示未卡壳，1.0 表示卡壳。
        """
        if self.mouse_collector is None or not self.mouse_collector.is_running():
            return 0.0
        tensor = self.mouse_collector.tensor()
        if tensor is None:
            return 0.0
        # tensor: [N,3] = (x, y, dt)
        max_dt = float(tensor[:, 2].max())
        displacement = float(np.hypot(tensor[-1, 0] - tensor[0, 0], tensor[-1, 1] - tensor[0, 1]))
        if max_dt > 2.5 or displacement < 0.02:
            return 1.0
        return 0.0

    def _dqn_decision(self, heuristic_delta: int) -> dict:
        """用 F5-1 DQN 根据当前状态决策难度调整方向。

        DQN 状态为 [连续答对数/10, 难度/5, 卡壳0/1]，动作 0=降难度 / 1=保持 / 2=升难度。
        模型不可用时回退到规则（heuristic_delta），保证主流程不中断。

        Args:
            heuristic_delta: 回退规则的难度增量（答对 +1 / 答错 -1）。

        Returns:
            {"delta", "action", "action_label", "state", "model"}。
        """
        state = np.asarray(
            [
                self.boss_state["consecutive_correct"] / 10.0,
                self.boss_state["difficulty"] / 5.0,
                self._dqn_is_stuck(),
            ],
            dtype=np.float32,
        )
        try:
            from modules.b7_dqn import api as dqn_api

            action = int(dqn_api.decide(state))
            delta = action - 1  # 0=降(-1) / 1=保持(0) / 2=升(+1)
            return {
                "delta": delta,
                "action": action,
                "action_label": DQN_ACTION_LABELS.get(action, "保持"),
                "state": state.tolist(),
                "model": "dqn",
            }
        except Exception as exc:
            logger.error("DQN 决策失败，回退规则：%s", exc)
            return {
                "delta": heuristic_delta,
                "action": heuristic_delta + 1,
                "action_label": DQN_ACTION_LABELS.get(heuristic_delta + 1, "保持"),
                "state": state.tolist(),
                "model": "rule",
            }

    def _boss_payload(self, decision: dict) -> dict:
        """组装 Boss 状态（含 DQN 决策信息）供前端展示。"""
        payload = dict(self.boss_state)
        payload.update(
            {
                "dqn_state": decision["state"],
                "dqn_action": decision["action"],
                "dqn_action_label": decision["action_label"],
                "dqn_model": decision["model"],
            }
        )
        return payload

    def _run_wrong_analyses(self, question: Question) -> dict:
        """执行错题专属分析（F4-3/F5-1/F6-1），逐项容错。

        Args:
            question: 触发分析的错题。

        Returns:
            各分析结果摘要。
        """
        analyses: dict = {}
        # F4-3 过度自信检测
        try:
            from modules.b5_mlp_confidence import api as confidence_api

            features = [
                question.confidence_self,
                question.mouse_hesitation_count,
                self.boss_state["difficulty"] / 5.0,
                min(question.time_cost / 60.0, 2.0),
            ]
            prob = confidence_api.confidence_probability(features)
            analyses["overconfidence"] = {
                "probability": round(prob, 3),
                "warn": prob > 0.7,
            }
        except Exception as exc:
            logger.error("过度自信检测失败：%s", exc)
            analyses["overconfidence"] = {"error": "模型未训练"}

        # F5-1 Boss 状态已由调用方更新
        analyses["boss"] = dict(self.boss_state)
        return analyses

    def _cluster_types(self) -> list:
        """执行题型聚类（F2-6）。

        Returns:
            聚类报告列表。
        """
        from modules.semantic import api as semantic_api

        reports = semantic_api.cluster_question_types(self.store.all())
        self._last_clusters = [r.__dict__ for r in reports]
        return self._last_clusters

    def clusters(self) -> list:
        """返回最近一次题型聚类结果。"""
        return getattr(self, "_last_clusters", [])

    def confusion_pairs(self) -> list:
        """返回易混淆概念对（F2-3）。"""
        from modules.semantic import api as semantic_api

        tags = self._collect_tags()
        if len(tags) < 2:
            return []
        return semantic_api.confusable_pairs(tags)

    def reading_history(self, limit: int = 50) -> list:
        """返回划词采集的历史记录（需开启划词采集）。

        Args:
            limit: 最多返回条数。

        Returns:
            历史划词记录列表。
        """
        if self.reading_collector is None:
            return []
        return self.reading_collector.history(limit)

    def record_reading(self, words: list, source: str, strategy: Optional[str] = None) -> bool:
        """记录一次划词序列（开关开启时落盘）。

        Args:
            words: 划选关键词序列。
            source: 场景（question/article）。
            strategy: 读题策略标签（可选）。

        Returns:
            是否成功记录（开关关闭时返回 False）。
        """
        if self.reading_collector is None or not self.reading_collector.is_running():
            return False
        return self.reading_collector.record(words, source, strategy)

    def _collect_tags(self) -> List[str]:
        """汇总全库知识点标签。"""
        tags: List[str] = []
        for q in self.store.all():
            tags.extend(q.knowledge_tags)
        return tags

    def shut_down(self) -> None:
        """关闭所有采集线程，释放资源。"""
        if self.mouse_collector is not None:
            self.mouse_collector.stop()
        if self.screen_collector is not None:
            self.screen_collector.stop()
        if self.reading_collector is not None:
            self.reading_collector.stop()
        logger.info("调度器已关闭所有采集线程")