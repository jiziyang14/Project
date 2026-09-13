# core（共享基础设施）

本目录职责：跨模块共享的协议、设备抽象、配置、存储与日志，是各子系统的唯一契约。
全局上下文见 ../../AI_CONTEXT.md 第 3 节「数据协议字典」。

| 文件 | 职责 |
|-|-|
| `schemas.py` | 数据协议 dataclass + 运行时类型/形状校验 |
| `device.py` | 设备抽象（CPU 默认，DML 经 `TRAE_DEVICE=dml` 可选启用） |
| `config.py` | JSON 配置加载、范围校验、相对路径构建 |
| `storage.py` | 本地题库/错题库存储（原子写） |
| `logger.py` | 统一日志（控制台 + 滚动文件） |

## 数据流

所有模块经 `core/schemas.py` 交换数据，避免跨模块结构漂移。

文章关键词重排与读题划词的特征口径分别固定在 `modules/semantic/keywords._to_features` 与 `huaci_candidates.huaci_to_features`（训练脚本在 `tools/`），训练与推理共用同一套特征，避免口径漂移（相关评估/训练：kw_eval.py、train_rerank.py、huaci_train.py、huaci_autolabel.py）。

## Quick Run

```bash
python -c "from core import device; print(device.describe())"
```