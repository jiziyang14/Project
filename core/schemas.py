"""数据协议契约层。

所有跨模块传递的数据结构定义为 dataclass，并在 __post_init__ 中做类型与形状严格校验，
确保畸形数据在第一时间被发现，避免传播至下游导致难以追溯的异常（Standards 第 8 节）。

本模块对应的数据结构已在 AI_CONTEXT.md 第 3 节「数据协议字典」中登记。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _require(condition: bool, message: str) -> None:
    """断言工具：校验失败即抛出带明确上下文的值错误。"""
    if not condition:
        raise ValueError(message)


@dataclass
class MousePoint:
    """鼠标轨迹中的单帧采样点。

    Args:
        x: 归一化横坐标，取值范围 [0, 1]。
        y: 归一化纵坐标，取值范围 [0, 1]。
        t: 距上一帧的时间差（秒），恒为非负。
    """
    x: float
    y: float
    t: float

    def __post_init__(self) -> None:
        _require(0.0 <= self.x <= 1.0, f"鼠标横坐标超出合法范围 [0,1]，得到 {self.x}")
        _require(0.0 <= self.y <= 1.0, f"鼠标纵坐标超出合法范围 [0,1]，得到 {self.y}")
        _require(self.t >= 0.0, f"时间差为负，非法：{self.t}")


@dataclass
class MouseTrajectory:
    """F1-1 鼠标轨迹采集的原始记录。

    滑窗 50 帧形成 [batch, 50, 3] 张量供 F3-1 Transformer 使用。
    """
    session_id: str
    timestamp: str
    trajectory: List[MousePoint]

    def __post_init__(self) -> None:
        _require(bool(self.session_id), "会话标识不能为空")
        _require(bool(self.timestamp), "时间戳不能为空")
        _require(len(self.trajectory) > 0, "轨迹帧数必须大于 0")


@dataclass
class ScreenshotFrame:
    """F1-2 屏幕截图的单帧元数据。

    图像张量 [3, 224, 224] 仅驻留内存队列流转，不落盘（隐私要求）。
    """
    frame_id: str
    timestamp: str
    tensor_id: str  # 内存队列中对应的缓存键

    def __post_init__(self) -> None:
        _require(bool(self.frame_id), "帧标识不能为空")
        _require(bool(self.tensor_id), "张量缓存键不能为空")


@dataclass
class Question:
    """F1-5 题目/错题的统一数据结构（见 Requirements 6.1）。

    答错的题即是错题（question 的子集），额外参与 F2-4/F4-2/F4-3/F5-1/F6-1。
    """
    question_id: str
    question_text: str
    options: List[str] = field(default_factory=list)
    correct_answer: str = ""
    user_answer: str = ""
    is_correct: bool = False
    confidence_self: int = 0  # 自评：0=蒙的，1=确定
    knowledge_tags: List[str] = field(default_factory=list)
    source: str = "manual"  # ocr | manual | photo
    time_cost: float = 0.0
    mouse_hesitation_count: int = 0
    created_at: str = ""
    is_mastered: bool = False
    is_wrong: bool = False
    # 该题的"该划关键词"：录入时自动生成＋人工增删，答题时据此做题面高亮、并用于评估学生划词质量
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require(bool(self.question_id), "题目标识不能为空")
        _require(bool(self.question_text), "题干不能为空")
        _require(self.confidence_self in (0, 1), f"自评非法，应为 0/1，得到 {self.confidence_self}")
        _require(self.source in ("ocr", "manual", "photo"), f"录入来源非法：{self.source}")
        if self.is_correct is True:
            _require(self.is_wrong is False, "答对与答错状态互斥，不能同时为真")


@dataclass
class ConfidenceFeatures:
    """F4-3 过度自信检测的输入特征（4 维）。

    字段顺序即模型输入顺序：自评、犹豫次数、难度值、耗时/平均耗时。
    """
    confidence_self: int
    hesitation_count: int
    difficulty_value: float
    time_ratio: float

    def to_vector(self) -> np.ndarray:
        """映射为 [4] 维浮点向量，顺序与模型输入约定一致。"""
        return np.asarray(
            [self.confidence_self, self.hesitation_count, self.difficulty_value, self.time_ratio],
            dtype=np.float32,
        )


@dataclass
class DifficultyFeatures:
    """F4-2 个性化难度预测的输入特征（3 维）。

    字段顺序即模型输入顺序：耗时/平均耗时、退格爆发次数、鼠标犹豫计数。
    """
    time_ratio: float
    backspace_burst_count: int
    mouse_hesitation_count: int

    def to_vector(self) -> np.ndarray:
        """映射为 [3] 维浮点向量，顺序与模型输入约定一致。"""
        return np.asarray(
            [self.time_ratio, self.backspace_burst_count, self.mouse_hesitation_count],
            dtype=np.float32,
        )


@dataclass
class DqnState:
    """F5-1 DQN 的三维状态，已归一化到 [0,1]。"""
    consecutive_correct: float  # 连续答对数 / 10
    difficulty_level: float       # 当前难度档位(1~5) / 5
    is_stuck: float               # 卡壳标签 0/1

    def to_vector(self) -> np.ndarray:
        """映射为 [3] 维浮点向量，order 固定为 (答对数, 难度, 卡壳)。"""
        return np.asarray(
            [self.consecutive_correct, self.difficulty_level, self.is_stuck],
            dtype=np.float32,
        )


@dataclass
class ClusterReport:
    """F2-6 题型聚类输出的一条记录。"""
    cluster_id: int
    cluster_label: str
    representative_questions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    percentage: float = 0.0
    question_ids: List[str] = field(default_factory=list)


@dataclass
class ErrorClusterReport:
    """F2-4 错因聚类输出的一条记录。"""
    cluster_id: int
    cluster_label: str
    percentage: float = 0.0
    wrong_answer_ids: List[str] = field(default_factory=list)


@dataclass
class GraphNode:
    """F6-1 二部图节点：左节点=知识点簇，右节点=行为标签。"""
    node_id: str
    kind: str  # "knowledge" | "behavior"
    label: str = ""


@dataclass
class CooccurrenceMatrix:
    """F6-1 知识点-行为共现矩阵的元数据载体。"""
    left_node_ids: List[str]
    right_node_ids: List[str]
    matrix: np.ndarray  # shape (len(left), len(right))，取值 0/1 或权重

    def __post_init__(self) -> None:
        expected = (len(self.left_node_ids), len(self.right_node_ids))
        _require(
            self.matrix.shape == expected,
            f"共现矩阵维度不匹配，期望 {expected}，得到 {self.matrix.shape}",
        )