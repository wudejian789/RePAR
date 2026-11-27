# HippoRAG 2

来源：[论文 2502.14802](https://arxiv.org/abs/2502.14802)，本目录 `paper.pdf`；[作者代码](https://github.com/OSU-NLP-Group/HippoRAG)。固定快照、逐文件哈希与 MIT 许可见 `sources.json`。使用当前锁定上游实现，不能把本地结果当作论文原配置复现。

## 封装范围

`HippoRetriever(context, client, encoder, working_dir=..., ...)` 保留官方 OpenIE、recognition memory 筛选、同义连接、passage/entity/fact 表示与 Personalized PageRank；只将原文和来源offset送入index，未送入问题、选项或答案。`retrieve` 返回原文片段、分数和offset。`close()` 关闭上游存储。顶层导出通用 `answer/retrieve_context`，LongBench 有独立 adapter。

本地注入 Qwen 和固定 MiniLM；每段240词、NER上限2048、triple上限4096，保留上游prompt和默认 recognition few-shot/DSPy示范，不在线训练DSPy或权重。调用串行执行，后端本地igraph/持久向量表，不需单独图数据库、商业API或专用GPU encoder。上游允许无可用过滤事实时回退dense检索；`stats.retrieval_mode`、`graph_seeds` 明确记录，不能把回退声称为PPR证据。自动抽取的三元组不等于已验证事实。

## 使用

```powershell
python -m baselines.HippoRAG2 --context knowledge.txt --task "research task" --config configs/repar_v2.json --encoder models/all-MiniLM-L6-v2 --working-dir artifacts/hippo_demo/index --retrieve-only
```

安装见 [总说明](../README.md)。索引目录绑定原文hash、完整模型配置、encoder revision、参数、适配代码及模型桥接代码哈希；拒绝不匹配或无身份的已有目录。不要拷贝manifest绕过检查。

构造参数 `workers=1` 保持普通入口默认串行；全量双服务实验显式 `workers=8`，只并行独立片段的作者OpenIE调用。并发客户端使用线程独立的真实响应元信息，索引身份同时绑定worker参数与客户端源码。实际图路径与dense fallback仍由作者查询结果记录，详见 [全量协议](../../docs/BASELINE_FULL_PROTOCOL.md)。

独立用例使用虚构人物/天文台，真实执行作者OpenIE解析、图构建、PPR与来源检查，模拟模型只用于机制测试。真实 Qwen pilot 的冷缓存、耗时、失败和图使用见 [筛选报告](../../docs/BASELINE_SCREENING.md)。额外embedding、固定示范和离线抽取开销单列，不能仅以相同最终LLM宣称全部资源公平。
