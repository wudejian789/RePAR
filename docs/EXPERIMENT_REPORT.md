# RePAR 完整实验交付

全部 **125 道 Multi-Document QA × 4 种方法 × 3 种部署 = 1,500 个预测**已完成，未用部分样本代替全量结果。
标准 Q4 的 RePAR 准确率为 **20.0%**，未超过论文 Table 3 的 GPT-4o 历史参考值 44.0%。
MTP 默认决策：**关闭**。同权重开启后的合成解码速度为关闭时的 **0.69×**，RePAR 服务计算时间变化为 **+4.2%**。

## 全量结果

| 部署 | 方法 | 正确/125 | 准确率 | 平均预填充秒 | 平均解码秒 | 平均服务计算秒 | 平均实测墙钟秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| 标准 Q4 | 直接回答 | 26 | 20.8% | 10.872 | 0.053 | 10.925 | 11.513 |
| 标准 Q4 | BM25 RAG | 29 | 23.2% | 0.414 | 0.034 | 0.448 | 0.566 |
| 标准 Q4 | RePAR | 25 | 20.0% | 1.429 | 3.355 | 4.784 | 5.069 |
| 标准 Q4 | RePAR 禁用图 | 24 | 19.2% | 1.429 | 3.357 | 4.785 | 0.323 |
| MTP 权重，推测关闭 | 直接回答 | 30 | 24.0% | 11.040 | 0.059 | 11.100 | 11.693 |
| MTP 权重，推测关闭 | BM25 RAG | 33 | 26.4% | 0.392 | 0.041 | 0.434 | 0.548 |
| MTP 权重，推测关闭 | RePAR | 28 | 22.4% | 1.454 | 3.359 | 4.813 | 5.112 |
| MTP 权重，推测关闭 | RePAR 禁用图 | 28 | 22.4% | 1.454 | 3.359 | 4.813 | 0.119 |
| MTP 权重，推测开启 | 直接回答 | 32 | 25.6% | 14.286 | 0.060 | 14.345 | 14.991 |
| MTP 权重，推测开启 | BM25 RAG | 32 | 25.6% | 0.485 | 0.043 | 0.528 | 0.645 |
| MTP 权重，推测开启 | RePAR | 27 | 21.6% | 1.707 | 3.310 | 5.017 | 5.327 |
| MTP 权重，推测开启 | RePAR 禁用图 | 27 | 21.6% | 1.707 | 3.312 | 5.020 | 0.150 |

![全部条件的准确率与计算成本](../artifacts/combined/comparison.png)

服务计算为每题所有 LLM 调用的 prompt_ms + predicted_ms 之和，覆盖预填充和解码。墙钟时间另含分词、索引加载、检索、HTTP 和本地缓存；禁用图条件可复用首次图调用前的相同 API 请求，墙钟值不能用来直接宣称其推理更快。服务计算与 tokens 对缓存调用仍计入首次真实推理成本。

## MTP：同权重开关的配对比较

两个 ModelScope 文件的 320 个共同张量中有 186 个不同，MTP 文件另有 15 个张量。因此标准文件与 MTP 文件的差异不能全部归于推测解码。此处仅比较同一份 MTP GGUF 的开关。
| 方法 | 准确率差 pp | 配对 bootstrap 95% CI | 胜/负 | 预测变化题数 | 服务计算加速比 | 加速比 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 直接回答 | +1.6 | [0.0, 4.0] | 2/0 | 2 | 0.77× | [0.77, 0.78] |
| BM25 RAG | -0.8 | [-2.4, 0.0] | 0/1 | 2 | 0.82× | [0.82, 0.83] |
| RePAR | -0.8 | [-5.6, 4.0] | 4/5 | 12 | 0.96× | [0.93, 0.99] |
| RePAR 禁用图 | -0.8 | [-5.6, 4.0] | 4/5 | 12 | 0.96× | [0.93, 0.99] |

配对重采样 20,000 次，seed=42；未作多重检验校正。相同 seed 不保证推测解码使用相同随机数路径，轨迹和预测可以变化。125 题不能证明严格的精度等价；置信区间宽度须结合差值看。RePAR 自适应轨迹可能不同，其服务时间比包含生成长度和取证路径变化。直接回答和 BM25 则保持输入文本一致。三种部署在同一 GPU 上按固定顺序运行，没有随机交错或多次全量重复；温度、动态频率和 Windows 桌面负载造成的系统性波动不会被题目 bootstrap 区间完全捕捉。

| 短上下文合成吞吐探针 | 解码 tokens/s |
|---|---:|
| 标准 Q4 | 346.8 |
| MTP 权重，推测关闭 | 370.0 |
| MTP 权重，推测开启 | 256.8 |

探针为 4 个固定输入 × 3 次重复，输出上限 320 tokens，统计丢弃第 1 轮后的 8 次调用。吞吐沿用 llama.cpp 的解码计数口径：每次 predicted_n 减去计入预填充的首 token，再除以总解码时间；原始输入、生成全文和 timings 均保存。它只度量短输入的长输出，不能替代 benchmark 的实际耗时。

| MTP 开启：真实 draft 统计 | 草稿 tokens | 接受 tokens | 接受率 |
|---|---:|---:|---:|
| 直接回答 | 1422 | 638 | 44.9% |
| BM25 RAG | 1380 | 645 | 46.7% |
| RePAR | 185304 | 90986 | 49.1% |
| RePAR 禁用图 | 185280 | 91007 | 49.1% |

接受率采用 API 的 draft_n_accepted / draft_n；服务日志还保存每次调用的接受统计。b12500 的 /props 中 speculative.types 是请求默认值，即使通过启动参数启用 MTP 也显示 none；实际启用以启动参数、creating MTP draft context 日志及非零 draft_n / draft_n_accepted 三者交叉验证。这里只测试最多 6 个 draft tokens 的固定配置，没有按测试集成绩搜索草稿长度或采样设置；不能将本配置结果外推为所有 MTP 部署的速度。

默认判据在汇总准确率前保存：相对标准 Q4 和同权重 MTP 关闭两个对照，direct 与 RePAR 均最多下降 1/125 题，RePAR 平均服务计算时间至少降低 20%，且合成解码吞吐至少提高 20%。这是工程选择规则，不是等价性检验。

```json
{
  "enable_mtp": false,
  "accuracy_operational_rule_met": true,
  "speed_operational_rule_met": false,
  "synthetic_decode_speedup": 0.6939047877249054,
  "synthetic_decode_speedup_vs_standard": 0.740366991776745,
  "repar_wall_speedup_vs_mtp_off": 0.9596014189594901,
  "repar_wall_speedup_vs_standard": 0.9515310487381946,
  "criterion": {
    "accuracy": "MTP on must be no more than 1 question lower than BOTH standard and MTP off on each of direct and RePAR; report paired intervals, not equivalence proof",
    "speed": "MTP on must have at least 20% lower mean summed server prefill+decode time for RePAR and at least 20% higher same-input synthetic decode throughput than BOTH standard and MTP off",
    "scope": "Operational rule only; 125 examples cannot establish a narrow statistical equivalence margin"
  },
  "note": "Accuracy rule is operational, not a statistical equivalence claim. Speed uses server computation, which is an end-to-end component rather than client wall time."
}
```

## 核心结构与实际运行

有序语义树保持连续原文范围，以段落/句子为优先边界；片段连接带字符出处的抽取式命题节点，图提供实体、概念、词汇关联及相邻片段桥接。tree_browse / tree_search / graph_search / read 都有可组合参数。控制器根据观察生成下一轮调用、反思和策略更新。算法细节见 [DESIGN.md](DESIGN.md)。
当前语义表示使用 TF-IDF，知识抽象使用抽取式命题。图边是相关证据的检索假设，并未验证为因果、矛盾或逻辑蕴含。任务内策略修订没有改变模型权重，也不代表跨任务 RSI。

| 部署 | 成功树查询 | 成功图查询 | 使用图的题 | ≥2 轮取证的题 | 重复调用错误 | 预算耗尽 | 回退题 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 标准 Q4 | 198 | 2 | 2 | 63 | 391 | 125 | 6 |
| MTP 权重，推测关闭 | 221 | 0 | 0 | 76 | 381 | 125 | 4 |
| MTP 权重，推测开启 | 209 | 0 | 0 | 67 | 396 | 125 | 3 |

反思字段和策略文本改变不等于有效自进化；上述调用统计用于判断 0.8B 控制器是否真正利用了图和多轮取证。若图访问稀少，禁用图消融不能证明图结构无用，只能反映本控制器的实际使用。重复调用会被拒绝并反馈；finish 与工具调用冲突时先执行工具；至少两轮取证后允许主动结束，最多 5 轮，耗尽预算照实记录。当前原型没有证明 RePAR 的精度收益，也没有实现超越历史大模型的研究目标。更详细的证据与改进假设见 [失败分析](FAILURE_ANALYSIS.md)。

## 数据、部署与公平性

数据为 ModelScope LongBench-v2 的全部 125 道 Multi-Document QA；索引只接触 context，不接触问题、选项或答案。问题和选项只用于在线查询；答案仅由评测器打分。没有根据测试集准确率选择提示或检索参数。JSON 协议预检使用了前两道 benchmark 的无标签输入，依据格式和重复调用问题做可靠性修正后固定配置；这属于协议适配时接触过测试输入，不能称为完全未见测试输入的预注册实验。协议预检、旧 27B 运行均归档，未混入此次 1,500 个结果。
三组都在同一张 RTX 3090 上顺序运行 llama.cpp b12500，Q4_K_M，262,144-token 窗口，parallel=1，seed=42，temperature=0.1，关闭 thinking。MTP 使用 draft-mtp、最多 6 个 draft tokens。当前模型不调用 9090 端口的原 27B 服务。
模型来源：[标准 GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF)、[MTP GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.5-0.8B-MTP-GGUF)；均选择 Q4_K_M。数据来源：[LongBench-v2](https://www.modelscope.cn/datasets/ZhipuAI/LongBench-v2)，历史对照：[论文 Table 3](https://arxiv.org/html/2412.15204v1#S4.T3)。下载的具体 revision、完整 SHA256 和每题上下文校验均保存在 artifacts 中。
直接回答有 111 题完整输入原文，14 题超过窗口，使用官方首尾截断规则。检索方法始终对完整原文构建索引。完整输入/截断子集的准确率在每部署详细报告中单列。
最终答案统一使用 GBNF 约束为 The correct answer is (A/B/C/D)。这避免 0.8B 在预算内未输出选项，但属于官方自由生成设置的明确偏离。RePAR 额外控制器计算不与直接回答等预算；输入/输出 tokens 和调用数均公开。论文 GPT-4o 44.0% 为 2024 年历史参考，没有在本机复测，也不能据此声称优于所有当前大模型。

## 原始材料与复现

- [标准 Q4 详细报告](REPORT_STANDARD.md)、[MTP 关闭详细报告](REPORT_MTP_OFF.md)、[MTP 开启详细报告](REPORT_MTP_ON.md)：分领域/长度/难度、配对统计、置信区间、成本、操作审计。
- `artifacts/combined/conditions.csv`：12 条件汇总；`all_predictions.csv`：逐题全部预测；`summary.json`：完整机器可读统计。
- `artifacts/experiment*/*.jsonl`：原始评测记录；`artifacts/traces_qwen08_*/`：每轮模型输出、调用参数、原文证据、反思与结束原因。
- `artifacts/cache/llm/`：内容寻址的原始请求/响应；`artifacts/deployment/`：模型版本、SHA256、张量比较、MTP 探针与 CUDA 日志；`artifacts/implementation_manifest.json`：推理代码/配置/提示校验。
- `artifacts/structure_audit.json`：全部 125 个知识库的字符出处、树覆盖、图端点及查询延迟审计。

```powershell
conda activate RePAR
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/download_assets.py
python scripts/build_indexes.py
python scripts/run_experiments.py
python scripts/finalize_results.py
```
