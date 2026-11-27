# RAPTOR / RaptorRAG

来源：[论文 2401.18059](https://arxiv.org/abs/2401.18059)，本目录 `paper.pdf`；[作者代码](https://github.com/parthsarthi03/raptor)，固定 commit 和 MIT 许可见 `sources.json`。

## 封装范围

`RaptorRetriever(context, client, encoder, ...)` 仅用知识原文建库，保留官方 UMAP 降维、soft GMM 聚类、递归摘要和 collapsed-tree 检索。`retrieve(query, top_k=24)` 返回原文/摘要及递归叶片 `source_spans`；摘要明确标为 `kind=summary`，不伪装原文。通用 `answer` 与 `retrieve_context` 从包顶层导出，LongBench 专用适配放在 `adapters/longbench.py`。

适配差异：用公共无损240词分段保留标点和offset；用配置中的 Qwen 生成100-token摘要；用本地冻结 MiniLM 替换论文/上游默认模型；最终统一为项目已有单轮 LongBench RAG 作答，未调用官方 QA 默认大模型。树深默认最多5层；保留上游小于等于11个节点不再建摘要层的停止条件，不为了轨迹强造树。当前依赖的 UMAP/GMM 仅设全局 NumPy seed，不能声称跨平台逐位确定性。

## 使用

```powershell
python -m baselines.RaptorRAG --context knowledge.txt --task "research task" --config configs/repar_v2.json --encoder models/all-MiniLM-L6-v2 --working-dir artifacts/raptor_demo/index --retrieve-only
```

安装与资源准备见 [baseline 总说明](../README.md)。不需训练或外部服务，但离线摘要和 embedding 计入成本；不能称作与仅词法建库的 RePAR 同资源。实现使用 hash-verified 作者快照，未修改第三方算法。索引当前在内存中构建，每次创建都会重建；HTTP缓存仍按真实请求身份复用。

验证包含独立构造的32片段实际 UMAP/GMM/摘要树及原文出处校验。真实 LongBench 限时结果、输入长度和中止条件见 [筛选报告](../../docs/BASELINE_SCREENING.md)，不是全量准确率结论。
