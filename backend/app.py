"""本地 HTTP 服务入口（FastAPI）。

仅监听 127.0.0.1，纯本地运行，前端通过同源请求调用。
启动：python -m backend.app
"""
from __future__ import annotations

import json

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.monitor import MonitorService
from backend.scheduler import Scheduler
from backend.training import router as train_router
from backend.reading_bank import router as reading_bank_router
from core.config import root_path
from core.logger import logger

app = FastAPI(title="AI 学习伴侣", docs_url=None, redoc_url=None)
app.include_router(train_router)
app.include_router(reading_bank_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = Scheduler()
monitor = MonitorService(scheduler)
insights = None  # 认知诊断结果缓存（每次请求按需重建）


# ── 前端静态资源 ──────────────────────────────
app.mount("/static", StaticFiles(directory=str(root_path("frontend"))), name="static")


@app.get("/")
def index():
    """返回前端首页。"""
    return FileResponse(str(root_path("frontend/index.html")))


# ── 题目录入 ─────────────────────────────────
def _auto_keywords(payload: dict) -> list:
    """录入题目时自动生成"该划关键词"候选，供人工确认/增删后随题保存。

    subject 影响划词策略（数学偏数字+单位/公式，语文偏考点/虚词/语病定位词），
    调用方未指定时默认语文。生成失败只记日志，不阻塞录入。
    """
    text = (payload.get("question_text") or "").strip()
    if not text:
        return []
    options = payload.get("options") or []
    subject = payload.get("subject") or "语文"
    try:
        from modules.semantic.huaci_candidates import generate_question_candidates
        cands = generate_question_candidates(text, options, subject=subject, top_k=12)
        return [c["kw"] for c in cands if c.get("kw")][:8]
    except Exception as exc:  # noqa: BLE001
        logger.error("自动生成题目关键词失败：%s", exc)
        return []


@app.post("/api/questions")
def add_question(payload: dict):
    """手动录入一道题（F0-2 / F1-5）。

    未显式提供 keywords 时，自动从题干+选项生成"该划关键词"候选；
    人工已给（含前端增删后回传）则原样保存。
    """
    keywords = payload.get("keywords")
    if not keywords:
        keywords = _auto_keywords(payload)
    question = scheduler.add_question(
        question_text=payload["question_text"],
        options=payload.get("options", []),
        correct_answer=payload.get("correct_answer", ""),
        knowledge_tags=payload.get("knowledge_tags", []),
        keywords=keywords,
        source=payload.get("source", "manual"),
        is_wrong=payload.get("is_wrong", False),
        question_id=payload.get("question_id"),
    )
    return {"question_id": question.question_id, "keywords": question.keywords, "count": scheduler.store.count()}


@app.get("/api/questions")
def list_questions():
    """返回全量题目（含错题）。"""
    return {"questions": [q.__dict__ for q in scheduler.store.all()]}


@app.post("/api/questions/suggest-keywords")
def suggest_keywords(payload: dict):
    """答题/校对时按需生成某题的"该划关键词"（题目未存 keywords 时即时算，不回写）。"""
    return {"keywords": _auto_keywords(payload)}


@app.post("/api/questions/ocr")
async def ocr_recognize(image: UploadFile = File(...)):
    """对上传图片做 OCR 识别（F0-1）。"""
    from modules.entry.ocr import OcrEngine

    import numpy as np

    data = await image.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    import cv2

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    engine = OcrEngine()
    if not engine.is_available():
        return {"ok": False, "error": "OCR 引擎不可用，请手动录入"}
    text = engine.recognize(img)
    parsed = engine.parse_question(text)
    return {"ok": True, "text": text, **parsed}


# ── 屏幕框选 OCR（后端抓屏，前端拖拽框选）────────────────
_CAPTURED_SCREEN = None  # 最近一次抓取的屏幕帧（BGR）


def _capture_screen():
    """抓取主屏幕，返回 BGR 图像。优先 mss，失败回退 PIL.ImageGrab。"""
    import cv2
    import numpy as np

    try:
        import mss

        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
    except ImportError:
        from PIL import ImageGrab

        return cv2.cvtColor(np.asarray(ImageGrab.grab()), cv2.COLOR_RGB2BGR)


@app.post("/api/screen/capture")
def screen_capture():
    """抓取主屏幕，返回预览图与尺寸，供前端框选。"""
    global _CAPTURED_SCREEN
    import base64
    import cv2

    try:
        img = _capture_screen()
    except Exception as exc:
        return {"ok": False, "error": f"屏幕捕获失败：{exc}"}
    _CAPTURED_SCREEN = img
    full_h, full_w = img.shape[:2]
    # 预览缩放到最长边 2400，前端按 scale 映射回全尺寸（提高清晰度，减少 4K 屏上放大发糊）
    scale = min(1.0, 2400.0 / max(full_w, full_h))
    preview = img if scale >= 1.0 else cv2.resize(img, (int(full_w * scale), int(full_h * scale)))
    ok, enc = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(enc.tobytes()).decode()
    return {
        "ok": True,
        "image": f"data:image/jpeg;base64,{b64}",
        "preview_w": int(preview.shape[1]),
        "preview_h": int(preview.shape[0]),
        "full_w": full_w,
        "full_h": full_h,
    }


@app.post("/api/screen/ocr")
def screen_ocr(payload: dict):
    """对已抓取的屏幕按框选区域裁剪并 OCR。"""
    global _CAPTURED_SCREEN
    if _CAPTURED_SCREEN is None:
        return {"ok": False, "error": "请先抓取屏幕"}
    try:
        x, y, w, h = int(payload["x"]), int(payload["y"]), int(payload["w"]), int(payload["h"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "框选参数无效"}
    img = _CAPTURED_SCREEN
    H, W = img.shape[:2]
    x1 = max(0, min(x, W))
    y1 = max(0, min(y, H))
    x2 = max(0, min(x + w, W))
    y2 = max(0, min(y + h, H))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return {"ok": False, "error": "框选区域无效"}
    from modules.entry.ocr import OcrEngine

    engine = OcrEngine()
    if not engine.is_available():
        return {"ok": False, "error": "OCR 引擎不可用，请手动录入"}
    text = engine.recognize(crop)
    parsed = engine.parse_question(text)
    return {"ok": True, "text": text, **parsed}


# ── 答题与错题判定 ──────────────────────────
@app.post("/api/answer")
def submit_answer(payload: dict):
    """提交答案并触发错题分析链（F4-3/F5-1）。"""
    result = scheduler.submit_answer(
        question_id=payload["question_id"],
        user_answer=payload["user_answer"],
        confidence_self=payload.get("confidence_self", 0),
    )
    return result


# ── 语义分析 ─────────────────────────────────
@app.get("/api/clusters")
def clusters():
    """题型聚类结果（F2-6），未缓存时按需计算。"""
    if not scheduler.clusters() and scheduler.store.count() >= 5:
        try:
            scheduler._cluster_types()
        except Exception as exc:
            logger.warning("聚类按需计算失败：%s", exc)
    return {"clusters": scheduler.clusters()}


@app.get("/api/confusion")
def confusion():
    """易混淆概念对（F2-3）。"""
    return {"pairs": scheduler.confusion_pairs()}


@app.post("/api/articles")
def article_keywords(payload: dict):
    """文章中心词提取（F2-5）：返回带流水线/模型/置信度的详细结果。"""
    from modules.semantic import keywords as kw

    try:
        result = kw.extract_keywords_full(payload["text"], top_n=payload.get("top_n", 8),
                                          title=payload.get("title") or "")
        return {"status": "ok", **result}
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:
        logger.error("文章中心词提取失败：%s", exc)
        return {"status": "error", "error": "中心词提取暂不可用"}


@app.post("/api/articles/stream")
def article_keywords_stream(payload: dict):
    """文章中心词提取（SSE 流式）：边处理边推送每阶段事件，用于实时过程可视化。

    事件格式（text/event-stream，每个块 data: json）：
      {"type":"stage","step":"文本预处理","items":N,"ms":xxx}
      {"type":"done","result":{...}}  或  {"type":"error","error":"..."}
    """
    import asyncio
    import json
    import queue
    import threading

    from fastapi.responses import StreamingResponse
    from modules.semantic import keywords as kw

    text = payload.get("text", "")
    if not text.strip():
        return {"status": "error", "error": "文章文本为空，无法提取中心词"}
    top_n = payload.get("top_n", 8)
    title = payload.get("title") or ""
    q: "queue.Queue" = queue.Queue()

    def emit(step, action, items, ms):
        q.put({"type": "stage", "step": step, "action": action, "items": items, "ms": ms})

    def worker():
        try:
            result = kw.extract_keywords_full(text, top_n=top_n, title=title, on_progress=emit)
            q.put({"type": "done", "result": {"status": "ok", **result}})
        except ValueError as exc:
            q.put({"type": "error", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.error("文章中心词流式提取失败：%s", exc)
            q.put({"type": "error", "error": "中心词提取暂不可用"})

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        try:
            while True:
                ev = await asyncio.to_thread(q.get)
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                if ev["type"] in ("done", "error"):
                    break
        finally:
            pass

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── P1 深度分析 ──────────────────────────────
@app.get("/api/error-clusters")
def error_clusters():
    """错因聚类（F2-4）：按错误答案语义分组。"""
    wrong = [q for q in scheduler.store.all() if q.is_wrong]
    try:
        from modules.semantic import api as semantic_api

        reports = semantic_api.cluster_wrong_answers(wrong)
        return {"clusters": [r.__dict__ for r in reports], "message": ""}
    except ValueError as exc:
        return {"clusters": [], "message": str(exc)}
    except Exception as exc:
        logger.error("错因聚类失败：%s", exc)
        return {"clusters": [], "message": "错因聚类暂不可用"}


@app.get("/api/questions/{question_id}/friction")
def question_friction(question_id: str):
    """认知摩擦分析（F2-1/F2-2）：线索词悬停权重 + 反常识词对。"""
    q = scheduler.store.get(question_id)
    if q is None:
        return {"error": "题目不存在"}
    from modules.semantic import api as semantic_api

    return {
        "question_id": question_id,
        "hesitation": semantic_api.hesitation_weights(q.question_text),
        "antonyms": [list(p) for p in semantic_api.antonym_pairs(q.question_text)],
    }


@app.post("/api/variant")
def generate_variant(payload: dict):
    """变体题生成（F5-2）：按领域感知规则生成，并返回改动说明。"""
    q = scheduler.store.get(payload["question_id"])
    if q is None:
        return {"error": "题目不存在"}
    from modules.intervention import variant_gen

    result = variant_gen.generate_variant(q, seed=payload.get("seed"))
    return {"question_id": q.question_id, **result}


@app.post("/api/variant/judge")
def judge_variant(payload: dict):
    """变体判别（F4-4）：评估变体是否值得推送。"""
    import numpy as np

    q = scheduler.store.get(payload["question_id"])
    if q is None:
        return {"error": "题目不存在"}
    variant_text = payload.get("variant_text", "")
    if not variant_text:
        return {"error": "缺少变体文本"}
    from modules.b6_mlp_variant import api as variant_api
    from modules.semantic import api as semantic_api

    orig_emb = semantic_api.embed_texts([q.question_text])[0]
    var_emb = semantic_api.embed_texts([variant_text])[0]
    diff = (orig_emb - var_emb).astype(np.float32)
    try:
        prob = float(variant_api.push_probability(diff))
        return {"push": prob > 0.6, "probability": round(prob, 3)}
    except Exception as exc:
        logger.error("变体判别失败：%s", exc)
        return {"error": "变体判别模型未训练"}


@app.post("/api/voice")
async def voice_analyze(audio: UploadFile = File(...)):
    """语音卡壳检测（F3-3）：解析上传的 WAV 并返回停顿/流畅度评估。"""
    from modules.voice import api as voice_api

    data = await audio.read()
    try:
        return voice_api.analyze_wav_bytes(data)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        logger.error("语音分析失败：%s", exc)
        return {"ok": False, "error": "语音分析暂不可用"}


@app.post("/api/reading")
def reading_capture(payload: dict):
    """F1-4 划词采集 + 分析：分析读题策略与语义向量，开关开启时落盘。"""
    words = payload.get("words") or []
    source = payload.get("source", "question")
    if not words:
        return {"error": "未记录到划词"}
    # 先分析（读题策略 + 语义向量）
    try:
        from modules.semantic import reading as reading_api

        result = reading_api.analyze_reading(words)
    except FileNotFoundError:
        result = {"error": "读题策略模型未训练"}
    except Exception as exc:
        logger.error("划词分析失败：%s", exc)
        result = {"error": "划词分析暂不可用"}
    # 采集落盘（需开启开关），带上策略标签供历史汇总
    recorded = scheduler.record_reading(words, source, strategy=result.get("strategy") if isinstance(result, dict) else None)
    if not recorded:
        return {"ok": False, "enabled": False, "message": "划词采集未开启，请到采集模块打开"}
    if "error" in result:
        return {"ok": False, "enabled": True, "error": result["error"]}
    return {"ok": True, "enabled": True, "recorded": True, **result}


@app.get("/api/reading/history")
def reading_history():
    """返回划词采集的历史记录（读题策略页汇总）。"""
    return {"records": scheduler.reading_history(limit=50)}


@app.post("/api/code/analyze")
def code_analyze(payload: dict):
    """编程刷题分析（F7-1 漂亮错误检测 + F7-2 压力测试建议）。"""
    code = payload.get("code") or ""
    if not code:
        return {"error": "请粘贴代码"}
    from modules.coding import api as coding_api

    return {
        "error_class": coding_api.classify_error(code, bool(payload.get("tle"))),
        "stress": coding_api.stress_profile(code),
    }


@app.get("/api/questions/{question_id}/difficulty")
def question_difficulty(question_id: str):
    """个性化难度预测（F4-2）。"""
    q = scheduler.store.get(question_id)
    if q is None:
        return {"error": "题目不存在"}
    from core.schemas import DifficultyFeatures
    from modules.b4_mlp_difficulty import api as diff_api

    features = DifficultyFeatures(
        time_ratio=min(q.time_cost / 60.0, 2.0),
        backspace_burst_count=q.mouse_hesitation_count,
        mouse_hesitation_count=q.mouse_hesitation_count,
    ).to_vector()
    try:
        value = float(diff_api.predict_difficulty(features))
        return {"difficulty": round(value, 3)}
    except Exception as exc:
        logger.error("难度预测失败：%s", exc)
        return {"error": "难度预测模型未训练"}


# ── 干预与融合 ───────────────────────────────
@app.get("/api/boss")
def boss():
    """RL Boss 状态（F5-1）：含 DQN 当前决策信息。"""
    payload = dict(scheduler.boss_state)
    try:
        decision = scheduler._dqn_decision(heuristic_delta=0)
        payload.update(
            {
                "dqn_state": decision["state"],
                "dqn_action": decision["action"],
                "dqn_action_label": decision["action_label"],
                "dqn_model": decision["model"],
            }
        )
    except Exception as exc:
        logger.error("Boss 状态 DQN 展示失败：%s", exc)
    return payload


@app.get("/api/graph")
def graph():
    """知识-行为关联图谱共现数据（F6-1）。

    左节点优先使用「题型聚类」（F2-6）产出的真实知识点簇标签，
    使图谱语义可读；聚类不足时回退为模型内置的通用簇名。
    """
    try:
        from modules.b8_graph import api as graph_api

        cm = graph_api.cooccurrence_matrix()
        left_nodes = list(cm.left_node_ids)

        # 用真实题型聚类标签替换通用「知识簇N」名，让用户看得懂
        try:
            clusters = scheduler.clusters()
            if clusters:
                labels = [(c.get("cluster_label") or c.get("cluster_id")) for c in clusters]
                left_nodes = [labels[i] if i < len(labels) else left_nodes[i] for i in range(len(left_nodes))]
        except Exception:
            pass  # 聚类不可用时保留通用名

        return {
            "ok": True,
            "left_nodes": left_nodes,
            "right_nodes": cm.right_node_ids,
            "matrix": cm.matrix.tolist(),  # numpy 转 list 以便 JSON 序列化
        }
    except Exception as exc:
        logger.error("图谱推理失败：%s", exc)
        return {"ok": False, "error": "认知图谱模型未训练或不可用"}


# ── 模块开关 ─────────────────────────────────
@app.get("/api/models/ready/{code}")
def model_ready(code: str):
    """检查指定模型子包是否已有训练权重（供训练状态页展示）。"""
    from core.config import root_path

    weights = list(root_path("data").glob(f"models/{code}/best_*.pt"))
    return {"code": code, "ready": bool(weights)}


@app.get("/api/modules")
def modules():
    """返回各采集模块开关状态。"""
    return scheduler.modules_enabled


@app.get("/api/monitor")
def monitor_state():
    """返回实时监控反馈状态（鼠标卡壳 + 截图注意力）。"""
    return monitor.state()


@app.get("/api/insights")
def cognitive_insights():
    """返回认知诊断摘要（难度/过度自信/错因/模型栈）。"""
    from backend.insights import build_insights

    return build_insights(scheduler.store, monitor.state())


@app.post("/api/modules/toggle")
def toggle_module(payload: dict):
    """切换采集模块开关。"""
    try:
        state = scheduler.toggle_module(payload["name"])
        return {"name": payload["name"], "enabled": state}
    except KeyError as exc:
        return {"error": str(exc)}


@app.on_event("startup")
def _startup() -> None:
    """启动时拉起实时监控后台线程，并后台预载语义模型（避免首次请求卡顿）。"""
    monitor.start()
    _preload_semantic_model()


def _preload_semantic_model() -> None:
    """后台线程预载多语言 SBERT 模型，首次做文章中心词时不再等待数秒模型加载。"""
    import threading

    def _load() -> None:
        try:
            from modules.semantic import sbert_service

            # 预热：加载多语言模型并跑一次空嵌入，触发权重落盘/进内存
            sbert_service.embed_multilingual(["预热"])
            logger.info("多语言语义模型已后台预载完成")
        except Exception as exc:
            logger.warning("语义模型后台预载失败（首次使用时将按需加载）：%s", exc)

    threading.Thread(target=_load, daemon=True, name="semantic-preload").start()


@app.on_event("shutdown")
def _shutdown() -> None:
    """关闭时回收采集线程与监控线程。"""
    monitor.stop()
    scheduler.shut_down()


if __name__ == "__main__":
    import os

    import uvicorn

    # 端口可配置：优先取环境变量 APP_PORT，其次 PORT，默认 8000，避免与其他服务冲突
    port = int(os.environ.get("APP_PORT", os.environ.get("PORT", "8000")))
    logger.info("本地服务启动：http://127.0.0.1:%d", port)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")