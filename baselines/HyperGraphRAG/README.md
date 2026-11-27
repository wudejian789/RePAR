# HyperGraphRAG

来源：[论文 2503.21322](https://arxiv.org/abs/2503.21322)，本目录 `paper.pdf`；[作者代码](https://github.com/LHRLAB/HyperGraphRAG)，固定 commit、SHA256 和 MIT 许可见 `sources.json`。

## 封装范围

`HyperGraphRetriever(context, client, encoder, working_dir=..., ...)` 调用作者 n-ary hyper-relation 抽取及 entity/hyperedge hybrid retrieval。使用上游 NetworkX 与 NanoVectorDB 本地存储，**不需要Neo4j服务器**。无超边时显式报错，不能把空图或普通向量检索冒充完整方法。

适配注入同一Qwen生成接口和冻结MiniLM，外层无损240词分段作为独立doc传给上游，内部1200-token分段、gleaning=2保留。抽取/中间调用输出上限2048；未为了赶时间取消gleaning。仅取作者 `only_need_context=True` 的真实检索bundle，统一使用项目最终作答prompt，不采用上游默认商业QA模型。实体/超关系描述是模型生成，bundle中显式区分于原文片段；`kind=upstream_context_bundle`，`provenance_store` 指向保存chunk来源的上游KV库。不是把整份bundle谎称原文。

## 使用

```powershell
python -m baselines.HyperGraphRAG --context knowledge.txt --task "research task" --config configs/repar_v2.json --encoder models/all-MiniLM-L6-v2 --working-dir artifacts/hyper_demo/index --retrieve-only
```

依赖和资源准备见 [总说明](../README.md)。索引身份绑定原文、模型、encoder、参数和适配代码，旧目录不兼容时请新建，不要覆盖历史索引。API为同步接口，延续上游event-loop行为；已有异步应用应在独立线程/进程运行同步CLI，不宣称原生异步并发支持。

构造参数 `workers=1` 保持原串行默认；双服务全量条件显式设8，将同步模型HTTP交给线程执行，避免阻塞作者抽取event loop。每片段的真实历史、gleaning次数与prompt不变；并发改变调度/可能影响插入顺序，单列新实验身份，详见 [全量协议](../../docs/BASELINE_FULL_PROTOCOL.md)。

独立虚构用例实际验证超边构建和带原文出处存储的hybrid检索。实际Qwen/LongBench计时、限额和失败记录见 [筛选报告](../../docs/BASELINE_SCREENING.md)。无需训练，但离线抽取/gleaning可能主导完整墙钟时间，需与在线检索分开报告。
