# 语义层（L1，含两个论文核心功能）

本目录职责：文章/题干等文本的语义理解与重点提取，即论文里的**两个核心功能**——
`文章关键词提取` 与 `读题划词`，以及 F2-1~F2-6 的聚类/混淆/摩擦分析。
全局上下文见 `../../AI_CONTEXT.md`。

| 文件/模块 | 职责 |
|-|-|
| `keywords.py` | **文章关键词提取**：TextRank+PMI 统计层候选 → MiniLM 语义层 → 多维特征梯度提升重排 → Top-N（含进度回调） |
| `huaci_candidates.py` | **读题划词候选生成 + 特征**：分学科识别候选，处理数字带单位/拉丁缩写/文言虚词/语病定位词，`huaci_to_features` 统一特征口径 |
| `huaci_predict.py` | **读题划词预测**：按学科加载 GradientBoosting，对候选打 0/1/2 分并排序 |
| `sbert_service.py` | 轻量语义模型加载（关键词语义层、混淆/聚类共用） |
| `confusion.py` | F2-3 易混淆语义推理 |
| `error_cluster.py` | F2-4 错误原因聚类 |
| `question_cluster.py` | F2-6 题型语义聚类（K-Means + 轮廓系数定 K） |
| `friction.py` | 认知摩擦 |
| `reading.py` | 读题划词采集/比对 |
| `api.py` | 对外隔离接口 |

## 数据流

- **文章关键词**：文章文本 → 统计层候选(Top-40) → 语义层精排 → 梯度提升重排 → Top-N 关键词（标题/词频/语义/位置等特征即时可算，不含标签）。
- **读题划词**：题干/选项 → 候选生成 → 分学科规则自动标注(0/1/2) → GradientBoosting 打分 → 建议划词。

## Quick Run

```bash
python -m tools.kw_eval        # 文章关键词独立测试集评估
python -m tools.huaci_train    # 读题划词分学科训练（输出 data/models/huaci_{数学,语文}.pkl）
```