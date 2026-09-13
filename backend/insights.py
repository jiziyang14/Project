"""认知诊断聚合：把多个深度学习模型的输出汇总为学习总览的"状态诊断"。

聚合来源：
- F2-4 错因聚类（语义层）
- F4-2 难度预测（b4 MLP）
- F4-3 过度自信检测（b5 MLP）
- 各模型就绪状态 + 实时监控输出（b1/b2）

所有子项逐项容错：单个模型不可用不影响整体，缺失项以 null/空 呈现。
"""
from __future__ import annotations

from typing import List

from core.config import root_path

# 模型栈清单（code, 业务语义名, 架构类型, 技术方案, 输入输出, 关键参数, 参数量）
MODEL_STACK = [
    ("b1_transformer", "鼠标卡壳预判", "Transformer", "Transformer 编码器", "输入 50×3 轨迹 / 输出 3 类", "d_model=64 · 层数 2", 110_000),
    ("b2_resnet", "截图注意力", "ResNet", "ResNet-18 + SimCLR", "输入 3×224×224 / 输出 64 维嵌入", "对比学习 · 余弦注意力", 11_200_000),
    ("b3_lstm", "读题策略", "LSTM", "LSTM 序列分类", "输入 划词 one-hot 序列 / 输出 3 类", "hidden=64 · 1 层", 200_000),
    ("b4_mlp_difficulty", "难度预测", "MLP", "MLP 回归", "输入 3 维特征 / 输出 难度 0~1", "3 层全连接", 81),
    ("b5_mlp_confidence", "过度自信", "MLP", "MLP 分类", "输入 4 维特征 / 输出 置信概率", "Sigmoid 二分类", 705),
    ("b6_mlp_variant", "变体判别", "MLP", "MLP 分类", "输入 384 维差异嵌入 / 输出 推送概率", "阈值 0.6", 12_353),
    ("b7_dqn", "RL Boss", "DQN", "DQN 强化学习", "状态 3 维 / 动作 3", "Q 网络 · ε-贪心", 227),
    ("b8_graph", "认知图谱", "Graph", "二部图嵌入", "知识点×行为 节点对打分", "内积 + Sigmoid", 2_000),
]


def _model_ready(code: str) -> bool:
    """检查模型子包是否已有最佳权重。"""
    return bool(list(root_path("data/models").glob(f"{code}/best_*.pt")))


def build_insights(store, monitor_state: dict) -> dict:
    """构建认知诊断摘要。

    Args:
        store: 题库存储。
        monitor_state: 实时监控状态（来自 MonitorService.state()）。

    Returns:
        供前端渲染的诊断摘要字典。
    """
    questions = store.all()
    total = len(questions)
    wrong = [q for q in questions if q.is_wrong]
    mastered = [q for q in questions if q.is_mastered]
    wrong_rate = len(wrong) / total if total else 0.0
    mastery_rate = len(mastered) / total if total else 0.0

    # 难度画像（F4-2）
    difficulty = {"mean": None, "level": None}
    mean_diff = _mean_difficulty(questions)
    if mean_diff is not None:
        difficulty["mean"] = round(mean_diff, 3)
        difficulty["level"] = "偏易" if mean_diff < 0.4 else "适中" if mean_diff < 0.7 else "偏难"

    # 过度自信风险（F4-3）
    over = _overconfidence_risk(wrong)

    # 错因分布（F2-4）
    clusters = _error_clusters(wrong)

    # 认知健康度：以未掌握占比为基准，存在过度自信时扣分
    health = round((1 - wrong_rate) * 100)
    if over["warn"]:
        health = max(0, health - 10)

    return {
        "health": health,
        "total": total,
        "wrong_rate": round(wrong_rate, 3),
        "mastery_rate": round(mastery_rate, 3),
        "difficulty": difficulty,
        "overconfidence": over,
        "error_clusters": clusters,
        "models": _model_stack(monitor_state),
    }


def _mean_difficulty(questions) -> float | None:
    """对题目计算平均个性化难度。"""
    try:
        from core.schemas import DifficultyFeatures
        from modules.b4_mlp_difficulty import api as diff_api
    except Exception:
        return None
    values = []
    for q in questions:
        if not q.question_text:
            continue
        features = DifficultyFeatures(
            time_ratio=min(q.time_cost / 60.0, 2.0),
            backspace_burst_count=q.mouse_hesitation_count,
            mouse_hesitation_count=q.mouse_hesitation_count,
        ).to_vector()
        try:
            values.append(float(diff_api.predict_difficulty(features)))
        except Exception:
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _overconfidence_risk(wrong) -> dict:
    """对错题样本评估过度自信风险。"""
    result = {"warn": False, "risk_count": 0, "total": len(wrong)}
    if not wrong:
        return result
    try:
        from modules.b5_mlp_confidence import api as conf_api
    except Exception:
        return result
    risk = 0
    for q in wrong:
        features = [q.confidence_self, q.mouse_hesitation_count, 0.5, min(q.time_cost / 60.0, 2.0)]
        try:
            if conf_api.confidence_probability(features) > 0.7:
                risk += 1
        except Exception:
            continue
    result["risk_count"] = risk
    result["warn"] = risk > 0
    return result


def _error_clusters(wrong) -> List[dict]:
    """错因聚类分布。"""
    if len(wrong) < 5:
        return []
    try:
        from modules.semantic import api as semantic_api

        reports = semantic_api.cluster_wrong_answers(wrong)
        return [
            {
                "label": r.cluster_label,
                "percentage": round(r.percentage, 3),
                "count": len(r.wrong_answer_ids),
            }
            for r in reports
        ]
    except Exception:
        return []


def _model_stack(monitor_state: dict) -> List[dict]:
    """模型栈：就绪状态 + 实时监控输出 + 参数量/架构类型。"""
    mouse_status = (monitor_state.get("mouse") or {}).get("status")
    screen_att = (monitor_state.get("screen") or {}).get("attention")
    stack = []
    for code, name, mtype, arch, io, detail, params in MODEL_STACK:
        status = ""
        if code == "b1_transformer":
            status = mouse_status or ""
        elif code == "b2_resnet":
            status = screen_att or ""
        stack.append({
            "code": code,
            "name": name,
            "type": mtype,
            "arch": arch,
            "io": io,
            "detail": detail,
            "params": params,
            "ready": _model_ready(code),
            "status": status,
        })
    return stack