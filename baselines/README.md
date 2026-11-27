# Baselines

这里存放与 RePAR 对比的独立方法实现。目录统一为 `baselines/<Method>/`；方法不放入 agent 核心代码、模型权重或数据集目录。

| 方法目录 | 实验方法 ID | 内容 |
|---|---|---|
| [BM25_RAG](BM25_RAG/README.md) | `bm25` | BM25 检索、一次基于原文的作答、自由任务入口与独立 benchmark 适配 |
| [RaptorRAG](RaptorRAG/README.md) | `RaptorRAG` | 官方 UMAP/GMM 递归摘要树，已封装，纳入限时 pilot |
| [HippoRAG2](HippoRAG2/README.md) | `HippoRAG2` | 官方 OpenIE、recognition filter、PPR，已封装，纳入限时 pilot |
| [HyperGraphRAG](HyperGraphRAG/README.md) | `HyperGraphRAG` | 官方超边抽取和 hybrid 检索，已封装，纳入限时 pilot |
| [RAGen](RAGen/README.md) | — | 数据生成与训练适配框架；论文已保存，未冒充单独检索算法 |
| [DO_RAG](DO_RAG/README.md) | — | 论文已保存，尚未核实作者代码，暂不实现 |
| [CoT_RAG](CoT_RAG/README.md) | — | 作者仓库已存档，但代码不完整且需领域决策树/案例条件，暂不封装 |
| [T3](T3/README.md) | — | 作者仓库已存档，外部教师 thinking traces 条件，暂不混入原文检索对照 |

## 新增方法的来源、依赖与试跑

七篇论文分别放在方法目录的 `paper.pdf`；`catalog.json` 是选型表，`sources.json` 保存论文 SHA256、作者仓库与固定 commit、源码逐文件哈希及许可信息。首轮选型和pilot见 [选型与效率报告](../docs/BASELINE_SCREENING.md)。用户随后授权三个方法继续完整125题，当前独立双副本并行运行，见 [全量协议](../docs/BASELINE_FULL_PROTOCOL.md)与[结果/状态](../docs/BASELINE_FULL_RESULTS.md)；启动不等于完成。

```powershell
python -m pip install -e ".[baselines,baseline-sources]"
python scripts/prepare_baseline_sources.py
python scripts/prepare_baseline_encoder.py
python -m baselines.RaptorRAG --context knowledge.txt --task "research task" --config configs/repar_v2.json --encoder models/all-MiniLM-L6-v2 --working-dir artifacts/my_raptor/index --retrieve-only
python scripts/run_baseline_pilot.py --config configs/baseline_pilot.json
```

`--retrieve-only` 只跳过最终作答；三种方法的离线摘要/抽取仍调用配置中的 LLM。普通任务会移除选择题答案 grammar，LongBench adapter 才恢复原有最终答案约束。默认单请求60秒、任务180秒是资源控制，不是固定检索路线。pilot 另用子进程保证硬超时；普通任务 CLI 的 deadline 为合作式，不能打断纯 CPU 聚类。

三种方法统一使用固定的 **all-MiniLM-L6-v2 CPU encoder**，权重与哈希在 `models/all-MiniLM-L6-v2/download_manifest.json`，不训练、不替换 Qwen；这与 RePAR 的词法表示不是同等辅助资源，也不是论文默认 encoder 的精确复现。长文本按该 encoder 的 tokenizer 分窗，窗口向量归一化后做不加权平均，再归一化，不静默丢弃超窗文本。完整实验如要区分结构收益与 encoder 收益，应增加相同 encoder 的 dense control。

源码快照位于 `third_party/baselines/<Method>/<commit>/`，保留官方算法而不修改快照。兼容层替换模型/向量接口、无损分段和存储位置；默认只装实际使用的本地存储依赖，不要求 Neo4j、外部向量服务、商业 API 或浏览器插件。不同方法的抽取调用数与预算需要单列，不能只比较最终一次生成。

wheel 包含封装、来源 manifest、adapter prompt，**不包含第三方源码、论文与权重**。从仓库外运行时指定绝对 `--config`、`--encoder`、`--working-dir` 路径，并设置 `$env:REPAR_REPOSITORY` 为含 `third_party/baselines` 的仓库根目录；可用 `$env:REPAR_BASELINE_REGISTRY` 指定来源表目录。准备脚本默认复用锁定 commit，显式 `--refresh-upstream` 才更新。CoT-RAG/T3 的固定快照未发现 LICENSE，开放可读不等于已获再分发许可，暂不分发或执行。

新增方法时延续以下布局，按需要增减文件：

```text
baselines/<Method>/
  __init__.py       对外导入接口
  retriever.py      该方法独有的检索逻辑
  pipeline.py       通用任务流程
  __main__.py       可选独立命令入口
  adapters/         特定 benchmark 的输入/输出适配
  README.md         算法、配置、运行与验证说明
```

公共文本处理、排序算子和模型客户端复用 `src/repar/` 对应模块，避免复制实现。baseline 本身不依赖 RePAR 的规划/反思流程；共用底层算子也不意味着共用 agent 工作流。测试按组件放在 `tests/` 下，实验脚本只做方法分发与编排。

Python 安装配置会包含 `baselines` 及其子包。新增方法后，应验证包导入、独立任务、所需评测适配与现有入口兼容性，不只验证从仓库根目录可运行。
