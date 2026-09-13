# AI_CONTEXT.md — 上下文持久记忆锚点

> 本文件是项目的**唯一全局认知锚点**。任何 AI 或开发者在会话中断、上下文清空后接手时，必须首先通读本文件才能开始工作。任何人在编写或修改任何模块前，必须先通读本文件（见 `Standards.md` 第 0 节）。

## 1. 系统冻结架构快照

### 1.1 项目定位
纯本地运行的 AI 学习伴侣，通过鼠标轨迹、屏幕截图等多维信号实时感知学习者认知状态，并提供个性化干预。**严禁调用任何云端/本地大语言模型 API，纯本地推理与训练。**

### 1.2 22 项原子功能（F1-1 ~ F7-2）

| 编号 | 功能 | 层 | 训练/推理 | 优先级 | 状态 |
|-|-|-|-|-|-|
| F1-1 | 鼠标轨迹采集 | L0 | 纯采集 | P0 | 已实现 |
| F1-2 | 屏幕截图采集 | L0 | 纯采集 | P0 | 已实现 |
| F1-3 | 麦克风录音 | L0 | 纯采集 | P2 | 已实现 |
| F1-4 | 读题划词顺序记录 | L0 | 纯采集 | P1 | 已实现 |
| F1-5 | 错题与自评入库 | L0 | 纯存储 | P0 | 已实现 |
| F0-1 | 框选OCR识别 | L0-Entry | 推理 | P0 | 已实现 |
| F0-2 | 手动粘贴录入 | L0-Entry | 纯表单 | P0 | 已实现 |
| F0-3 | 拍照上传录入 | L0-Entry | 推理 | P1 | 已实现 |
| F2-1 | 线索词悬停权重统计 | L1 | 纯规则 | P1 | 已实现 |
| F2-2 | 反常识词对挖掘 | L1 | 纯规则 | P1 | 已实现 |
| F2-3 | 易混淆语义推理 | L1 | 预训练推理 | P0 | 已实现 |
| F2-4 | 错误原因聚类 | L1 | 预训练推理+聚类 | P1 | 已实现 |
| F2-5 | 文章中心词提取 | L1 | 预训练推理 | P0 | 已实现 |
| F2-6 | 题型语义聚类 | L1 | 预训练推理+聚类 | P0 | 已实现 |
| F3-1 | 鼠标走神预判 | L2 | **训练+推理** | P0 | 已实现 |
| F3-2 | 截图注意力监测 | L2 | **训练+推理** | P0 | 已实现 |
| F3-3 | 语音卡壳检测 | L2 | 预训练推理 | P2 | 已接入Whisper(+启发式兜底) |
| F4-1 | 读题策略分类 | L3-Diag | **训练+推理** | P1 | 已实现 |
| F4-2 | 个性化难度预测 | L3-Diag | **训练+推理** | P1 | 已实现 |
| F4-3 | 过度自信检测 | L3-Diag | **训练+推理** | P0 | 已实现 |
| F4-4 | 变体题有效性判别 | L3-Diag | **训练+推理** | P1 | 已实现 |
| F5-1 | RL Boss动态出题 | L3-Interv | **训练+推理** | P0 | 已实现 |
| F5-2 | 变体题自动生成 | L3-Interv | 纯规则 | P1 | 已实现 |
| F5-3 | 游戏化界面 | L3-Interv | 纯展示 | P1 | 已实现 |
| F6-1 | 知识点-行为关联图谱 | L4 | **训练+推理** | P0 | 已实现 |
| F7-1 | 漂亮错误检测 | L5 | 预训练推理 | P2 | 已接入SBERT(+规则兜底) |
| F7-2 | 编程压力测试 | L5 | 纯规则 | P2 | 已实现 |

### 1.3 8 个深度学习模型清单（可作为独立子包）

| 模型子包路径 | 对应功能 | 类型 | 输入张量 | 输出 |
|-|-|-|-|-|
| `modules/b1_transformer/` | F3-1 | Transformer Encoder | `[batch,50,3]` | 3类(流畅/犹豫/卡壳) |
| `modules/b2_resnet/` | F3-2 | ResNet-18+SimCLR | `[batch,3,224,224]` | 特征嵌入128维 |
| `modules/b3_lstm/` | F4-1 | LSTM | 划词one-hot序列 | 3类(数据/逻辑/跳跃) |
| `modules/b4_mlp_difficulty/` | F4-2 | MLP回归 | 3维特征 | 难度值0~1 |
| `modules/b5_mlp_confidence/` | F4-3 | MLP分类 | 4维特征 | 盲目自信概率 |
| `modules/b6_mlp_variant/` | F4-4 | MLP分类 | 384维差异嵌入 | 推送概率 |
| `modules/b7_dqn/` | F5-1 | DQN | 3维状态 | 3动作(降/保/升) |
| `modules/b8_graph/` | F6-1 | 二部图嵌入 | 节点对 | 共现概率 |

### 1.4 已排除技术栈（绝对红线）
- ❌ 任何大语言模型（LLM）
- ❌ 云端 API 调用（OpenAI / HuggingFace Inference / 在线知识库）
- ✅ 允许一次性下载预训练权重（SBERT / ResNet / Whisper），下载后推理完全本地
- ✅ 纯本地推理与训练，数据仅存本地文件系统

## 2. 关键决策日志

| 日期 | 决策 | 理由 |
|-|-|-|
| 2026-08-14 | 训练后端为 **CPU 默认 + DML 可选**（`TRAE_DEVICE=dml` 显式启用） | 实测 torch-directml 对 conv2d/embedding/Transformer 编码层等关键算子直接崩溃（INTERNAL ASSERT），无法作为可靠默认；CPU 保证所有功能完整，DML 仅用于纯线性模型的显式加速 |
| 2026-08-14 | 开发节奏：先 P0 垂直切片 | 规模庞大，先跑通核心闭环降低风险 |
| 2026-08-14 | 允许一次性下载预训练权重 | 文档要求推理不联网，一次性权重获取符合初衷，网络可用 |
| 2026-08-14 | 配置文件统一用 JSON 格式 | 遵循 Standards.md 第 10 节（与 Requirements 示例的 yaml 冲突时以 Standards 为准） |
| 2026-08-14 | F4-1 读题策略 LSTM 落地训练：现实化合成生成器 + 真实数据积累接口 | 无真实标注数据，用带噪声/模式漂移的生成器造训练集（300 合成样本，评估准确率 0.933），并新增 `add_real_sample/load_real_samples`（JSONL 落盘 `data/reading_strategy_real.jsonl`）供后期积累真实读题数据自动并入训练集 |
| 2026-08-14 | F5-1 RL Boss 真正接入答题闭环：难度决策由 DQN 驱动 | 此前 `/api/boss` 仅返回静态 boss_state，DQN 未被调用；现 `scheduler.submit_answer` 用 DQN 状态 `[连续答对数/10, 难度/5, 卡壳0/1]` 决策难度调整方向（0降/1保/2升），模型不可用时回退规则，`/api/boss` 与 Boss 卡片均展示 DQN 决策与状态 |
| 2026-08-14 | 划词采集开关语义确认：采集模块独立开关，关闭时静默不采集 | 开关关闭时前端完全静默、不落盘、不提示；开启后划词才触发读题策略 LSTM 分类 + SBERT 语义向量采集并写入历史 |
| 2026-08-14 | F3-3 语音卡壳接入 Whisper，保留能量启发式兜底 | 用 faster-whisper（tiny，CPU int8，本地）转写，从转写时间戳推导卡壳并输出转写文本；Whisper 未装/加载失败/音频超长时自动回退能量启发式，`model` 字段区分 whisper/heuristic，前端展示转写文本与来源 |
| 2026-08-14 | F7-1 漂亮错误检测接入 SBERT，保留规则兜底 | 实测 SBERT 对代码判型不可靠（会把创新型/常规型判反、置信度≈0.02），故标签由规则字典决定（可靠），SBERT 仅输出语义置信度（confidence）作为汇报增强；模型字段 rule+sbert / rule |
| 2026-08-30 | 读题划词判定口径＝「学生划词 vs 该题保存的标准关键词」，输出覆盖/命中/漏划 | 放弃旧"数据/逻辑优先"读题策略三分类；做题划词准确性直接比对标准词更直观可解释 |
| 2026-08-30 | 前端"答题 / 审题"双模式合一（作答与审题共用一套读题划词与讲评逻辑） | 审题=只读题划词不写答案，练读题抓重点；统一组件，改动同源生效 |
| 2026-08-30 | 划词标准关键词在录入/编辑题目时「一键划词」生成并持久保存到该题；做题据此高亮并判划词准确性 | 需提前确定标准词，避免做题时现场生成导致口径漂移 |
| 2026-08-30 | 论文结构定为「叙事三要素」：引言 / 主要成果论述(交互+技术+实验) / 结论；多级自动编号已修复（一级=章0级/二级=1级/三级=2级，共用一个 multilevel 实例） | 对齐官方"引言+主要成果论述+结论"三要素要求，并保证章节级联编号正确 |
| 2026-08-30 | 量化口径按实际数据/模型缓存核实更正：划词训练样本 10493（数学5821/语文4672）、holdout 数学0.938/语文0.803；关键词独立27篇 Hit@1 89%(规则81%)、Gold 召回 0.75(0.70) | 部分旧文档数字（1026题/10403样本/0.95-1.0/0.86）已过时，论文须以实跑/缓存值为准 |

## 3. 数据协议字典

所有结构在 `core/schemas.py` 中定义为 `dataclass` 并做类型/形状校验，跨模块传递必须使用。

| 数据结构 | 字段 | 类型 | 合法范围 | 来源模块 |
|-|-|-|-|-|
| `MouseTrajectory` | session_id, timestamp, trajectory | str, str, list[Point] | 50帧 `[batch,50,3]`(x,y,Δt) | F1-1 |
| `ScreenshotFrame` | frame_id, timestamp, tensor_id | str, str, str | 224×224×3 内存流转不落盘 | F1-2 |
| `Question` | question_id, text, options, correct, user_answer, ... | dataclass | 见 6.1 | F0-1/2/3, F1-5 |
| `MouseTensor` | batch | Tensor `[batch,50,3]` | x,y∈[0,1],Δt≥0 | F1-1→F3-1 |
| `ImageBatch` | batch | Tensor `[batch,3,224,224]` | 0~1 | F1-2→F3-2 |
| `ClusterReport` | cluster_id, label, keywords, percentage, question_ids | dataclass | 由聚类产生 | F2-6 |
| `ErrorClusterReport` | cluster_id, label, percentage, wrong_answer_ids | dataclass | 错题≥5触发 | F2-4 |
| `DqnState` | consecutive_correct, difficulty_level, is_stuck | list | 归一化 0~1 | F5-1 |
| `CooccurrenceMatrix` | left/right node ids, matrix | np.ndarray | 二部图 | F6-1 |

## 4. 环境速查

- **Python**: 3.10.11
- **设备**: Intel Core Ultra 9 185H（含 NPU）、32GB RAM、Intel Arc Graphics（无 NVIDIA GPU）
- **训练后端**: CPU 默认，DML 经 `TRAE_DEVICE=dml` 可选启用（`core/device.py` 统一抽象）
- **numpy**: 必须 <2（paddlepaddle 2.6 与 numpy 2.x ABI 不兼容）
- **托管源注意**: tsinghua PyPI 镜像缺部分包（python-multipart/numpy1.26/paddleocr），需从 pypi.org 或 Paddle 官方源安装；HuggingFace 需可直连（hf-mirror.com 亦可）
- **关键受训版本**: torch 2.4.1（torch-directml 强锁）、sentence-transformers 3.0.1 + transformers 4.57.6（须兼容 torch 2.4.1，禁用 transformers>=5）
- **首要依赖**: torch, torch-directml, sentence-transformers, keybert, scikit-learn, paddleocr, fastapi, uvicorn, typeguard
- **OCR**: PaddleOCR（PP-OCRv3，本地离线，可选启用；加载失败时手动录入仍可用）

## 5. 目录职责索引

见根目录 `README.md`。所有一级目录 README 首段必须引用本文件的相对路径。