# CoT-RAG（作者资料已存档，暂不封装）

[论文 2504.13534](https://arxiv.org/abs/2504.13534)，本目录 `paper.pdf`；论文明确链接 [hustlfy123/CoT-RAG](https://github.com/hustlfy123/CoT-RAG)。来源清单与固定commit见 `sources.json`，源码在对应 `third_party/baselines/CoT_RAG/`。本次先后遇到GitHub API限流，已用同commit官方归档完成源码获取；错误与恢复保留在来源记录。

它并非普通“CoT提示+RAG”：主方案由专家给出粗粒度领域决策树，LLM细化为图，结合知识案例/子案例检索和伪程序执行。论文另有 **Zero-expert** 消融，不能说所有变体都必须人工专家，也不能把这个变体与主方案混报。

当前公开快照的main仍引用未定义的 `DecisionTree`、大小写不一致的 `LLMAdapter`，并示意性使用领域初始树；不是接收任意长文就能完整运行的通用实现。若为LongBench临时设计领域树/案例，需要额外协议和独立验证，不能用测试答案补专家知识。固定快照未见LICENSE，暂不再分发或运行。

因此本轮保留论文、真实代码和适配边界，未提供误导性的空壳运行接口；可在单独专家/案例库条件或明确Zero-expert复现协议下后续评审。见 [选型报告](../../docs/BASELINE_SCREENING.md)。
