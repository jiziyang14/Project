# backend（本地 HTTP 服务 + 调度器）

本目录职责：对外提供 FastAPI HTTP 服务，并串联 P0 核心闭环。
全局上下文见 ../../AI_CONTEXT.md。

| 文件 | 职责 |
|-|-|
| `app.py` | FastAPI 路由，仅监听 127.0.0.1，服务前端静态资源 |
| `scheduler.py` | 流程调度器：题库、采集器生命周期、答题判定、错题分析链 |
| `monitor.py` | 实时监控反馈服务：鼠标卡壳(F3-1)/截图注意力(F3-2) 后台推理，GET /api/monitor |
| `insights.py` | 认知诊断聚合（难度/过度自信/错因/模型栈），GET /api/insights |
| `training.py` | 关键词标注/训练/评估服务，路由 /api/train/*（文章关键词 + 读题划词） |
| `reading_bank.py` | 审题题库服务（题源 data/readingbank/{物理,语文}.jsonl），路由 /api/readingbank/* |

## 数据流

```
前端 → /api/questions        → 录入 → 题库
前端 → /api/answer           → 判定 → F4-3/F5-1 错题分析
前端 → /api/clusters         → F2-6 题型聚类
前端 → /api/confusion        → F2-3 易混淆
前端 → /api/articles         → F2-5 中心词
前端 → /api/articles/stream  → F2-5 中心词（SSE 流式过程可视化）
前端 → /api/graph            → F6-1 关联图谱
前端 → /api/modules/toggle   → 采集模块开关
前端 → /api/readingbank/random → 审题题库抽题（带标准划词，供审题对照）
前端 → /api/train/*          → 文章关键词/读题划词 标注·训练·评估工作台
```

## Quick Run

```bash
python -m backend.app   # http://127.0.0.1:8000
```