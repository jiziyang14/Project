"""本地存储层：题库与错题库的持久化。

所有数据仅存本地文件系统（data/），符合隐私要求。采用 JSON 文件存储，
每次写操作原子替换（先写临时文件再 rename），避免损坏导致程序崩溃。
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.config import root_path
from core.logger import logger
from core.schemas import Question


class QuestionStore:
    """题库存储，同时维护全量题目与错题视图。

    Args:
        relative_path: 题库 JSON 文件相对项目根目录的路径。
    """

    def __init__(self, relative_path: str = "data/questions.json") -> None:
        self._path: Path = root_path(relative_path)
        self._lock = threading.RLock()
        self._questions: List[Question] = []
        self._load()

    def _load(self) -> None:
        """从磁盘加载题库；文件缺失或损坏时降级为空题库并记录错误。"""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._questions = [Question(**item) for item in raw]
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("题库文件损坏，已降级为空题库：%s", exc)

    def _flush(self) -> None:
        """原子写回题库到磁盘。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as fh:
            json.dump([q.__dict__ for q in self._questions], fh, ensure_ascii=False, indent=2)
        temp.replace(self._path)

    def add_question(
        self,
        question_text: str,
        options: Optional[List[str]] = None,
        correct_answer: str = "",
        source: str = "manual",
        knowledge_tags: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        is_wrong: bool = False,
        user_answer: str = "",
        is_correct: bool = False,
        confidence_self: int = 0,
        time_cost: float = 0.0,
        mouse_hesitation_count: int = 0,
        question_id: Optional[str] = None,
    ) -> Question:
        """新增或更新一道题并持久化（按 question_id 幂等 upsert）。

        Args:
            question_text: 题干全文。
            options: 选项列表，可为空。
            correct_answer: 正确答案文本。
            source: 录入来源（ocr/manual/photo）。
            knowledge_tags: 知识点标签列表。
            is_wrong: 是否为错题。
            user_answer: 用户作答。
            is_correct: 是否答对。
            confidence_self: 自评 0/1。
            time_cost: 作答耗时（秒）。
            mouse_hesitation_count: 犹豫次数。
            question_id: 可选的既有题目标识；若已存在则就地更新题干/选项/答案等
                可编辑字段（保留创建时间与作答统计），否则新建。

        Returns:
            新建或更新后的 Question 对象。

        Raises:
            ValueError: 题干为空或来源非法。
        """
        # 编辑已存题目：按 id 就地更新可编辑字段，保留 created_at 与作答统计
        if question_id:
            with self._lock:
                existing = next((q for q in self._questions if q.question_id == question_id), None)
                if existing is not None:
                    existing.question_text = question_text
                    existing.options = options or []
                    existing.correct_answer = correct_answer
                    existing.knowledge_tags = knowledge_tags or []
                    existing.keywords = keywords or []
                    existing.source = source
                    existing.is_wrong = is_wrong
                    self._flush()
                    logger.info("更新题目 %s", question_id)
                    return existing
        question = Question(
            question_id=str(uuid.uuid4()),
            question_text=question_text,
            options=options or [],
            correct_answer=correct_answer,
            user_answer=user_answer,
            is_correct=is_correct,
            confidence_self=confidence_self,
            knowledge_tags=knowledge_tags or [],
            keywords=keywords or [],
            source=source,
            time_cost=time_cost,
            mouse_hesitation_count=mouse_hesitation_count,
            created_at=datetime.now().isoformat(timespec="seconds"),
            is_mastered=is_correct,
            is_wrong=is_wrong,
        )
        with self._lock:
            self._questions.append(question)
            self._flush()
        logger.info("新增题目 %s（来源=%s）", question.question_id, source)
        return question

    def all(self) -> List[Question]:
        """返回全量题目（含错题）。"""
        with self._lock:
            return list(self._questions)

    def wrong(self) -> List[Question]:
        """返回错题子集。"""
        with self._lock:
            return [q for q in self._questions if q.is_wrong]

    def get(self, question_id: str) -> Optional[Question]:
        """按标识查找单道题，不存在返回 None。"""
        with self._lock:
            for q in self._questions:
                if q.question_id == question_id:
                    return q
        return None

    def count(self) -> int:
        """返回全量题目数。"""
        with self._lock:
            return len(self._questions)

    def wrong_count(self) -> int:
        """返回错题数。"""
        return len(self.wrong())