# 工具脚本集

本目录职责：离线数据处理、模型训练、测试集构建、报告中台文件读取等辅助脚本，不对外提供服务。
全局上下文见 ../AI_CONTEXT.md。

> 注意：`_deps/` 为本地内置的第三方依赖（PIL / docx / lxml），仅作离线回退，不属于业务模块。

## 脚本一览

| 脚本 | 功能 |
|-|-|
| `fetch_corpus.py` | 公开语料抓取（供标注训练数据，选用公有文本规避版权） |
| `import_cjeval.py` | C-Eval 题目导入（初中/高中物理、语文等） |
| `build_reading_bank.py` | 审题题库构建（data/readingbank/{物理,语文}.jsonl） |
| `huaci_build_data.py` | 划词训练数据构建 |
| `huaci_gen_pool.py` | 划词候选池生成 |
| `huaci_autolabel.py` | 划词候选自动标注 |
| `huaci_train.py` | 划词模型训练 |
| `huaci_eval.py` | 划词模型评估 |
| `train_rerank.py` | 关键词重排模型训练 |
| `train_export.py` | 训练结果导出 |
| `kw_eval.py` | 关键词抽取评估 |
| `_build_testset.py` / `_build_independent_testset.py` | 测试集构建（独立集用于脱离训练的公平评估） |
| `_gen_testset_pool.py` | 测试集候选池生成 |
| `_clean_testset.py` | 测试集去重清洗 |
| `_diag_testset.py` | 测试集诊断 |
| `_leak_check_testset_articles.py` | 训练/测试语料泄露检查 |
| `_read_progress_xlsx.py` / `_read_report_pptx.py` / `_read_report_docx.py` | 读取报告中台（Excel/PPT/Word）文件 |

## Quick Run

```bash
# 依赖均已在项目根 requirements.txt 中
python -c "import tools; print('tools 可导入')"
```