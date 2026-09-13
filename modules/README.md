# modules（功能实现）

本目录职责：22 项功能（F1-1 ~ F7-2）的具体实现，含 8 个可训练深度学习模型子包。
全局上下文见 ../AI_CONTEXT.md。

| 子目录 | 对应功能 | 说明 |
|-|-|-|
| `b1_transformer/` | F3-1 | 鼠标卡壳 Transformer（训练+推理） |
| `b2_resnet/` | F3-2 | 截图注意力 ResNet+SimCLR（训练+推理） |
| `b3_lstm/` | F4-1 | 读题策略 LSTM（P1） |
| `b4_mlp_difficulty/` | F4-2 | 难度预测 MLP（P1） |
| `b5_mlp_confidence/` | F4-3 | 过度自信 MLP（训练+推理） |
| `b6_mlp_variant/` | F4-4 | 变体判别 MLP（P1） |
| `b7_dqn/` | F5-1 | RL Boss DQN（训练+推理） |
| `b8_graph/` | F6-1 | 二部图嵌入（训练+推理） |
| `semantic/` | F2-1~2-6 + 关键词/划词 | 语义层：关键词抽取/重排（keywords.py）、划词候选生成与预测（huaci_candidates.py / huaci_predict.py）、SBERT 服务、聚类 |
| `entry/` | F0-1~0-3 | 录入层（OCR 等） |
| `collect/` | F1-1~1-4 | 采集层 |

## 规范

- 每个 DL 子包含 `train.py / infer.py / config.json / model.py / dataset.py / api.py / test.py / README.md`。
- 子包间经 `api.py` + `core/schemas.py` 隔离，单模块异常不波及其他。

## Quick Run

```bash
python -m modules.b1_transformer.test
python -m modules.b5_mlp_confidence.train --mock
```