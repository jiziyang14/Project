"""实时监控反馈服务：周期消费采集数据，跑深度学习推理并缓存最新状态。

闭环链路：
  采集器(mouse/screen) → MonitorService 周期取数 → F3-1/F3-2 推理 → 内存状态
  → GET /api/monitor → 前端状态灯

采集器由用户显式开启（默认关）。只有当对应采集开启且模型已训练时才做推理，
否则返回可读的提示（采集未开启 / 模型未训练 / 数据不足），保证前端永远有明确反馈。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

from core.config import root_path
from core.logger import logger

# 推理周期（秒）。采集频率远高于此，推理取最近窗口即可。
INTERVAL = 2.0
# 截图相似度保留的历史点数，用于判断注意力漂移
SCREEN_HISTORY = 40
# 注意力判定阈值：最近若干帧平均相似度低于该值视为"漂移"
ATTENTION_THRESHOLD = 0.8


def _now() -> str:
    """返回当前 UTC 时间的 ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _model_ready(code: str) -> bool:
    """检查指定模型子包是否已有最佳权重（磁盘 glob，开销小）。"""
    return bool(list(root_path("data/models").glob(f"{code}/best_*.pt")))


class MonitorService:
    """实时监控服务的运行期状态与后台推理线程。

    Args:
        scheduler: 持有采集器实例的调度器。
        interval: 推理周期（秒）。
    """

    def __init__(self, scheduler, interval: float = INTERVAL) -> None:
        self.scheduler = scheduler
        self.interval = interval
        self._screen_history: deque = deque(maxlen=SCREEN_HISTORY)
        self._state: dict = self._empty_state()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _empty_state() -> dict:
        """返回初始空状态（JSON 可序列化）。"""
        return {
            "mouse": {
                "enabled": False,
                "model_ready": False,
                "status": None,      # 流畅 / 犹豫 / 卡壳
                "probs": None,       # [p流畅, p犹豫, p卡壳]
                "note": "",
                "updated_at": None,
            },
            "screen": {
                "enabled": False,
                "model_ready": False,
                "similarity": None,  # 相邻帧余弦相似度
                "attention": None,   # 专注 / 漂移
                "history": [],       # 最近若干帧相似度
                "note": "",
                "updated_at": None,
            },
        }

    # ── 生命周期 ──────────────────────────────
    def start(self) -> None:
        """启动后台推理线程（幂等）。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("实时监控反馈服务已启动（周期 %.1fs）", self.interval)

    def stop(self) -> None:
        """停止后台推理线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("实时监控反馈服务已停止")

    def _loop(self) -> None:
        """后台循环：周期更新鼠标与截图监控状态。"""
        while self._running:
            try:
                self._update_mouse()
                self._update_screen()
            except Exception as exc:
                logger.error("监控更新失败：%s", exc)
            time.sleep(self.interval)

    def state(self) -> dict:
        """返回当前监控状态（供接口直出 JSON）。"""
        return self._state

    # ── 鼠标：F3-1 卡壳三分类 ──────────────────
    def _update_mouse(self) -> None:
        s = self._state["mouse"]
        collector = self.scheduler.mouse_collector
        enabled = collector is not None and collector.is_running()
        s["enabled"] = enabled
        if not enabled:
            s.update(status=None, probs=None, note="采集未开启", updated_at=None)
            return
        if not _model_ready("b1_transformer"):
            s.update(model_ready=False, status=None, probs=None, note="模型未训练", updated_at=None)
            return
        s["model_ready"] = True
        tensor = collector.tensor()  # (50, 3)，窗口未满为 None
        if tensor is None:
            s.update(status=None, probs=None, note="轨迹数据不足（需 50 帧）", updated_at=None)
            return
        try:
            from modules.b1_transformer import api as b1

            probs, labels = b1.classify(torch.as_tensor(tensor).unsqueeze(0))
            p = [round(float(v), 3) for v in probs[0]]
            idx = int(np.argmax(p))
            s.update(
                status=labels[idx],
                probs=p,
                note=f"输入 50帧×3通道 · 置信度 {p[idx]:.2f}",
                updated_at=_now(),
            )
        except Exception as exc:
            logger.error("鼠标推理失败：%s", exc)
            s.update(status=None, probs=None, note="推理暂不可用", updated_at=None)

    # ── 截图：F3-2 注意力相似度 ────────────────
    def _update_screen(self) -> None:
        s = self._state["screen"]
        collector = self.scheduler.screen_collector
        enabled = collector is not None and collector.is_running()
        s["enabled"] = enabled
        if not enabled:
            s.update(similarity=None, attention=None, note="采集未开启", updated_at=None)
            return
        if not _model_ready("b2_resnet"):
            s.update(model_ready=False, similarity=None, attention=None, note="模型未训练", updated_at=None)
            return
        s["model_ready"] = True
        batch = collector.as_batch(n=2)  # (2,3,224,224)，缓存为空为 None
        if batch is None:
            s.update(similarity=None, attention=None, note="截图数据不足", updated_at=None)
            return
        try:
            from modules.b2_resnet import api as b2

            emb = b2.embed_batch(torch.as_tensor(batch))
            sim = float(torch.dot(emb[0], emb[1]).item())
            self._screen_history.append(sim)
            recent = list(self._screen_history)[-5:] or [sim]
            avg = float(np.mean(recent))
            attention = "专注" if avg >= ATTENTION_THRESHOLD else "漂移"
            s.update(
                similarity=round(sim, 3),
                attention=attention,
                history=[round(float(v), 3) for v in self._screen_history],
                note=f"输入 224×224 截图对 · 近5帧均值 {avg:.2f}",
                updated_at=_now(),
            )
        except Exception as exc:
            logger.error("截图推理失败：%s", exc)
            s.update(similarity=None, attention=None, note="推理暂不可用", updated_at=None)