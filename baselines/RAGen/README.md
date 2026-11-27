# RAGen（资料占位，未实现）

[Domain-Specific Data Generation Framework for RAG Adaptation](https://arxiv.org/abs/2510.11217)。已保存 `paper.pdf`，实际PDF版本和SHA以 `sources.json` 为准；不要与同名RL agent训练项目RAGEN混淆。

它生成 Question–Answer–Context 训练信号，并演示 embedding 对比微调与LLM监督微调；不是能独立替换普通检索器的完整单轮推理baseline。本次没有定位到可核实的作者官方GitHub链接，未用同名第三方项目代替，未编写猜测式复现。

可以作为**单列训练数据/适配条件**：需指定不含测试答案的训练知识库、数据生成教师、训练算力和下游retriever/reader，再比较。不能把训练后LLM结果标为相同未训练Qwen的training-free对照。本轮不运行。后续获得作者代码或明确训练协议后再实现，见 [选型报告](../../docs/BASELINE_SCREENING.md)。
