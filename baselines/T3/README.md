# T3（作者资料已存档，独立辅助语料条件）

[RAG over Thinking Traces Can Improve Reasoning Tasks](https://arxiv.org/abs/2605.03344)，本目录 `paper.pdf`；[作者代码](https://github.com/Narabzad/t3)，固定commit与文件SHA见 `sources.json`。

T3先用强推理模型在独立辅助题集生成thinking traces，再离线转换为结构化、反思式或语义变换表示，在线检索后供solver推理。solver权重无需训练，但教师生成、trace语料与转换模型属于额外资源。其原始目标以数学、知识推理、代码任务为主，与从LongBench给定长文检索原文不是相同知识输入。

可以做单列 **external-trace** 条件，前提是辅助题集、教师、去污染和成本均明确，绝不使用LongBench测试答案生成trace。当前不把普通文段摘要检索改名为T3；未复制或执行上游预计算测试检索结果。固定快照未见LICENSE，暂不再分发或运行。仓库/论文结果不完全一致，本地不引用其数字作为复测结果。

本次仅保存论文和作者源码，不编写缺少合法trace语料时仍貌似成功的fallback。见 [选型报告](../../docs/BASELINE_SCREENING.md)。
