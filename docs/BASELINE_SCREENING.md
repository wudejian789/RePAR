# 七种 baseline 选型与效率筛选

核对日期：2026-11-18，Asia/Shanghai。此文记录本地适配和小规模效率筛选，完整125题LongBench v2实验仍等待用户review。历史RePAR成绩没有重跑、覆盖或作为本轮新增实测。

## 比较边界与选择

| 方法 | 作者GitHub | 训练/额外资源 | 本轮处理 |
|---|---|---|---|
| RAPTOR | parthsarthi03/raptor | 无新增权重训练；冻结encoder、离线摘要 | 官方算法封装，效率pilot |
| HippoRAG2 | OSU-NLP-Group/HippoRAG | 无本地权重训练；encoder、OpenIE、固定recognition few-shot | 官方算法封装，效率pilot |
| HyperGraphRAG | LHRLAB/HyperGraphRAG | 无新增权重训练；encoder、超关系抽取/gleaning | 官方算法封装，效率pilot |
| RAGen | 未定位到可核实作者仓库 | QAC数据生成，embedding对比微调/LLM监督微调的适配条件 | 论文与来源占位，未猜测实现 |
| DO-RAG | 未定位到可核实作者仓库 | 多模态预处理、领域KG、pgvector/PostgreSQL等；实际训练依赖未复现 | 论文与来源占位，未猜测实现 |
| CoT-RAG | hustlfy123/CoT-RAG | 主方案专家决策树与案例库；另有Zero-expert消融 | 作者源码已存档，但入口仍含未定义类/占位内容；暂不包装为可运行通用baseline |
| T3 | Narabzad/t3 | solver无需微调，但增加教师生成的外部thinking traces及转换成本 | 作者源码已存档，独立辅助语料条件，未混入原文检索主对照 |

以上取舍依据原论文与固定源码，而非根据LongBench答对与否选择。RAGen不是同名RAGEN强化学习框架；CoT-RAG不是任意CoT+RAG；T3不是将原文摘要改名为thinking trace。没有官方链接只表示本次未核实，不能据此断言不存在开源。CoT-RAG/T3固定快照未见LICENSE，当前不再分发或执行。

七篇PDF均保存在 `baselines/<Method>/paper.pdf`，逐篇标题已核对；论文SHA、上游commit、文件SHA和许可在各目录 `sources.json`。代码原样快照在 `third_party/baselines`，适配不修改快照。CoT-RAG下载曾遇GitHub元数据API限流，随后同commit官方archive下载成功，保留恢复记录。DO-RAG已核对为 arXiv:2505.17058；论文没有GitHub链接。

三种主候选适用于训练自由的结构化检索对照，但不能称作全部资源相同：RePAR现有表示以词法/英语spaCy为主，新方法增加冻结MiniLM。完整实验应至少单列离线建库、encoder和生成成本，并考虑加入同encoder的dense control，才可区分结构与表示收益。当前不是论文公布模型与超参的精确复现，不引用论文成绩作为本地成绩。

最终reader prompt和解码相同不代表证据预算完全相同：top_k=24在不同API中分别作用于片段、树节点或实体/超边，RAPTOR另有8000 cl100k-token上限，Hyper保留作者bundle预算。应将本轮理解为显式配置下的系统可行性比较。若要严格控制最终输入token，需另设同Qwen-token预算条件，不能不披露地截短作者方法的结构证据。

## 实现与协议

- 三个方法目录均提供 `Retriever(context, client, encoder, ...)` 风格的具体类、`retrieve`、通用 `retrieve_context/answer` 与 `python -m baselines.<Method>`；benchmark逻辑在独立adapter。
- 上游默认远程模型改为配置中的 Qwen3.5-0.8B-Q4_K_M，deployment为 `qwen35-08b-q4km-standard-b12500`，temperature=0.1、seed=42、关闭thinking和MTP。用户文字中的“0.58”与实际配置不符，本轮沿用已核验的0.8B服务。
- 三者统一使用 all-MiniLM-L6-v2，revision为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，CPU运行，额外权重约91MB；逐文件SHA在模型manifest。长文本按原tokenizer窗口分别编码，等权平均窗口向量后归一化，避免静默截断。
- 原文统一无损240词分段；问题/选项只在查询时传入，标准答案仅在最终作答后评分。树/图构建接口不接受问题、选项和答案。
- RAPTOR保留UMAP/soft GMM/递归摘要/折叠树检索，摘要100 tokens、最多5层、检索上下文8000 cl100k tokens；原文与摘要分别标记并回溯叶片offset。UMAP版本和随机实现有跨环境非确定性，不能声称逐位复现。
- HippoRAG2保留OpenIE、默认recognition示范、同义边与PPR，NER2048/triple4096 tokens；上游dense fallback由stats单列，不能声称每次都使用图。
- HyperGraphRAG保留超关系抽取、gleaning=2与hybrid检索，抽取/中间输出上限2048；仅用官方本地NetworkX+NanoVectorDB，无需Neo4j/商业API/浏览器插件。bundle标为派生证据，保留chunk出处库。
- 索引和中间调用剥离最终答案grammar；最终LongBench作答沿用已有官方0shot_rag模板、答案grammar及128-token上限。保留官方算法检索bundle，最终reader统一，不声称复现各论文原reader。
- Hippo/Hyper持久索引绑定context、完整解码配置、encoder revision、上游commit、适配与模型桥接源码hash，拒绝不兼容或身份不明的旧目录。RAPTOR每次在内存重建。原历史请求缓存、预测与冻结manifest未改写。

## Pilot选择与资源门槛

协议在推理前保存于 `artifacts/baseline_pilot_20261118_v1/protocol.json`，附模型配置、完整数据SHA、样本context SHA、执行源码快照及来源manifest。

从既有125题 Multi-Document QA 的short/medium/long三档，各选**词数最少**的一题，以ID作同词数tie-break；没有看标签挑题，全文建库，不剪短上下文。此设计给出乐观可行性下界，不能估算整档平均效率或全量准确率。

| 长度档 | ID | 原文词数 |
|---|---|---:|
| short | 66f37eb9821e116aacb2d295 | 8,205 |
| medium | 66faa8efbb02136c067c7357 | 33,209 |
| long | 66fb6527bb02136c067c7b4f | 131,841 |

运行顺序BM25→RAPTOR→Hippo→Hyper，单GPU服务串行；每方法从短到长，每题完整子进程**180秒**硬限额，单HTTP请求60秒且不重试，合作式deadline提前5秒以保存结果。任一题未完成就不继续该方法更长样本，门槛与答案无关，未运行条目明确标记而非算作超时。每次使用新的索引和请求缓存；服务prompt cache未重置，不能声称冷模型启动或无服务器前缀复用。

完整墙钟包括Python/依赖启动、encoder加载、建库、检索和最终作答；另存各phase、embedding时间、模型调用、token、finish_reason、HTTP错误、cache hit与失败原始日志。独立fixture测试使用模拟输出只检验机制，以下效率必须来自真实Qwen服务。

## 实测结果

数据来源为 `artifacts/baseline_pilot_20261118_v1/summary.json`，每题有 `result.json`、`stdout.log`、`calls.jsonl` 和真实请求缓存。已执行8个条件：6个完成，2个在预算临界处建库失败；另4个按预定门槛未运行。冷缓存指新的请求缓存/索引，不表示清空系统页缓存、Numba JIT缓存或重启模型。

| 方法 | short完整墙钟 | medium完整墙钟 | long完整墙钟 | 当前建议 |
|---|---:|---:|---:|---|
| BM25（重跑control） | 5.8秒 | 5.9秒 | 5.9秒 | 保留既有低成本参照 |
| RAPTOR | 31.6秒 | 60.2秒 | 134.4秒 | **三个样本均通过180秒门槛，优先列入用户review名单** |
| HippoRAG2 | 179.5秒，建库失败 | 未运行 | 未运行 | 当前底座/预算下暂不继续全量 |
| HyperGraphRAG | 179.2秒，建库失败 | 未运行 | 未运行 | 当前底座/预算下暂不继续全量 |

RAPTOR三题的纯建库时间为21.8/50.2/125.0秒，encoder加载约4.7–5.0秒，检索0.030/0.091/0.075秒，最终生成0.362/0.481/0.254秒，其余为进程和依赖启动等开销。分别产生7/46/141个摘要节点，实际检索选中的摘要为6/6/15个，确实使用了抽象层。三题总共194次摘要生成都到达预定100-token上限，不能说摘要均自然完整结束；这是保留上游摘要预算的本地配置，未据答案调大预算。

HippoRAG2在最短short样本已完成51个LLM响应，输出59,295 tokens，其中16次达到长度上限（NER 5次2048、triple 11次4096）；剩余预算不足，最终上游报告25个chunk的triple extraction失败，没有完成检索或作答。HyperGraphRAG在同一原文完成37个响应、输出55,316 tokens，其中27次达到2048上限；建库末次HTTP请求在剩余约3秒时ReadTimeout，未进入检索/作答。两个条件各记录1次HTTP错误。它们是**在当前0.8B生成行为与明确预算下不实用**，不能据此断言算法在更强模型或预建共享知识库下必然慢，也不能把不完整索引算作成功。

所有288个已完成响应均无客户端请求缓存命中，真实请求身份、模型/解码参数及最终6次答案grammar检查通过；两项失败保留完整日志，4项门槛跳过不冒充超时。当前执行过的39份冻结源码均与protocol哈希一致。仓库中随后补充的每方法adapter只是委托到同一已冻结公共adapter，未改变试跑算法或用答案调参。

作为审计披露，BM25三题2/3正确、RAPTOR三题1/3正确；3题有意选的是各档最短上下文，样本太小且不随机，**不能当作可靠准确率估计或据此排名**。预测/标签原样保留，没有因为结果不利删除案例。本轮接受/排除只依据预先声明的完成状态与时限。

可review的新增主候选是 **RAPTOR**；HippoRAG2与HyperGraphRAG封装仍保留可运行入口和通用机制测试，当前不继续完整benchmark。RAPTOR也只通过三个乐观下界样本：该领域最长上下文约404,556词，不能保证所有125题都满足180秒。完整实验若获用户确认，应先明确是否沿用门槛、如何计入完整离线成本和失败，以及是否补同encoder dense control，不能事后删慢题换取完整覆盖名义。

## 验证与可复核文件

- 独立虚构案例验证了实际UMAP/GMM摘要树、Hippo OpenIE/PPR、HyperGraph超边/hybrid与来源；模拟输出仅用于机制测试，不混入真实pilot。
- 普通和隔离环境的 `python -m pytest tests webserver/tests -q` 均为33项通过，无skip；仓库外wheel安装及隔离Python导入、同prompt、三个来源快照加载、真实CPU encoder超窗输入分成3窗均通过。
- 严格隔离检查发现filelock/jinja2/mpmath/networkx原先来自用户site；已将同版本安装进RePAR环境，并补齐其依赖，隔离 `pip check` 通过，未改已完成pilot的请求或模型配置。
- 来源验证逐篇核对7篇PDF和239份上游源码/配置/许可文件的SHA；无权重训练，未下载上游测试集预测或外部trace来作答。
- 审计与环境文件在 `artifacts/maintenance/baseline_pilot_audit.json`、`baseline_source_validation.json`、`baseline_wheel_validation.json`、`baseline_environment_20261118.json`，最终测试输出在 `baseline_pytest_20261118.txt`。前端代码未改，本次未额外手工操作浏览器或生成新页面截图。

## 后续边界

本轮不启动完整benchmark，不依据3题准确率调prompt、改检索参数或挑有利样本。已运行样本如后续用于调试，必须披露其已见属性；本轮没有把测试答案用于核心机制修改。全量实验需要新标识、冻结代码/配置和完整125唯一ID检查，不能续写进pilot或覆盖旧结果。
