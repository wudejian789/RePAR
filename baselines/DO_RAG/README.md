# DO-RAG（资料占位，未实现）

[DO-RAG: A Domain-Specific QA Framework Using Knowledge Graph-Enhanced Retrieval-Augmented Generation](https://arxiv.org/abs/2505.17058)。`paper.pdf`已核对标题与作者，来源和SHA在 `sources.json`。

论文描述多模态文档处理、多层知识图、语义检索融合及grounded refinement，提及pgvector/PostgreSQL等组件。方法本身可作为领域KG-RAG对照，但原论文领域数据/外部服务条件不能默认为LongBench纯文本同资源条件。论文叙述以推理流水线为主；本次没有复现证据支持“所有部件无需训练/外部依赖”的保证。

截至本次资料核对，没有定位到可核实的作者代码，论文文本也未给出GitHub链接；**这不等同于断言互联网上不存在代码**。按用户要求暂不实现，不用自写KG-RAG冒充DO-RAG。保留论文和目录，后续获得明确官方代码再决定适配。
