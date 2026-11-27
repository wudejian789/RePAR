# BM25 RAG

通用单轮 baseline：按原文段落/句子边界切分片段，以 BM25 排序取 top-k，将选中的片段按来源顺序交给模型作答。无需构建树、实体图，也不运行 RePAR 的规划或反思循环。

## 独立使用

在安装项目的 Python 环境中，纯检索不需要模型服务：

```powershell
python -m baselines.BM25_RAG --context knowledge.txt --task "research task" --retrieve-only
```

生成自由形式回答时通过 JSON 配置提供模型连接信息：

```powershell
python -m baselines.BM25_RAG --context knowledge.txt --task "research task" --config configs/repar_v2.json --top-k 24 --output artifacts/bm25_task.json
```

该命令只读取模型连接/解码设置，不启用配置中的树、图或控制器。自由任务不会使用 benchmark 的选项字母输出约束。`--chunk-words` 默认 240，`--top-k` 默认 24；返回证据包含片段 ID、准确原文、字符区间和 BM25 分数。

Python 接口：

```python
from baselines.BM25_RAG import BM25Retriever, answer, retrieve_context

source = BM25Retriever(context, chunk_words=240)
evidence_text, records = retrieve_context(task, source, top_k=8)
result = answer(task, source, client, top_k=8, max_tokens=512)
```

`client` 提供与项目模型客户端一致的 `complete(messages, max_tokens, tag=...)` 接口。核心 pipeline 不接收标准答案，也不处理选项或评分。

## 代码边界

| 文件/模块 | 作用 |
|---|---|
| `retriever.py` | 原文片段和 BM25 倒排索引；不引用树/图构建器 |
| `pipeline.py` | 检索原文与自由形式 RAG 作答 |
| `adapters/longbench.py` | 原有 LongBench prompt、问题/选项拼接与答案解析适配 |
| `__main__.py` | 普通知识文本上的独立命令 |
| `repar.text` | 公共分词、无损片段划分 |
| `repar.retrieval.lexical` | 与 RePAR 树检索共用的 BM25 排序算子，非 baseline 工作流 |

共享排序保留原参数 `k1=1.5`、`b=0.75`、query term frequency 上限 2，以及同分时按来源位置排序的行为。

## 现有实验兼容

原实验命令仍使用 `--method bm25`；评测入口会直接分发到本方法，历史 agent 的同名分支只做兼容委托。原问题/选项 query、检索片段、官方作答 prompt、`bm25_answer` 调用标签和结果字段保持兼容。

独立索引的构建统计仅反映实际 BM25 构建成本，不再包含未使用的树/图成本。历史原始记录保留其当时统计，不因目录迁移而重写。

验证：

```powershell
python -m pytest tests/test_bm25_rag.py tests/test_index.py -q
python scripts/validate_baseline_migration.py
```

后一个命令使用已有125题的输入、预测与原始请求做离线兼容性核对，不调用模型或重新评分。
