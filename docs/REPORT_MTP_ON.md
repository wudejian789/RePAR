# RePAR：LongBench v2 Multi-Doc QA 完整实验

125/125 题、四个预先固定条件均已完成。RePAR 准确率 **21.6%**，未达到数值上超过论文 Table 3 GPT-4o 44.0% 的目标。
这是与论文历史结果的数值比较；没有重新调用 GPT-4o，不能据此宣称优于所有当前大模型。

## 设置

数据来自 ModelScope ZhipuAI/LongBench-v2，503 题中全部 125 道 Multi-Document QA；SHA-256、每题 ID 和上下文校验见 `artifacts/dataset_manifest.json`。
子领域：Governmental 23、Academic 50、Financial 15、Multi-news 23、Legal 14；长度：short 58、medium 44、long 23。
本地 RTX 3090 部署 Qwen3.5-0.8B-MTP-Q4_K_M，Q4_K-Medium，262,144-token 服务窗口，接口 http://127.0.0.1:9091。所有控制、检索编排与作答均使用这个 0.8B 底座，不调用原 9090 服务或更大模型。部署参数、GGUF 校验和及 CUDA 日志见 `artifacts/deployment/`。
temperature=0.1、seed=42、enable_thinking=false；直接回答和最终答案最多 128 输出 tokens。所有条件的最终答案采用相同 GBNF 约束，只能输出 The correct answer is (A/B/C/D) 的合法句式，防止模型先解释导致 128-token 预算内未作答。该格式约束是对官方评测的明确偏离，不能称作逐字复现官方解码设置。RePAR 每轮控制器最多 768 tokens、最多 5 轮、每轮最多 3 个工具调用，证据记忆最多 10,000 words。
不进行参数训练；本项目 Parametric Retrieval 指查询 API 的参数化。索引完全由 context 构建，不接触 question、choices、answer；查询可使用问题与四个选项。
自进化指单题内查询策略、计划和证据记忆的反馈修正；没有模型权重自修改或跨题使用答案的训练。代码与配置在汇总打分前固定，见 `artifacts/implementation_manifest.json`。

## 主结果

| 方法 | 正确/总数 | 准确率 | Wilson 95% CI | 无效答案 | 平均 LLM 调用 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 0.8B 直接回答 | 32/125 | 25.6% | [18.8, 33.9] | 0 | 1.00 |
| Qwen3.5 0.8B + BM25 RAG | 32/125 | 25.6% | [18.8, 33.9] | 0 | 1.00 |
| Qwen3.5 0.8B + RePAR | 27/125 | 21.6% | [15.3, 29.6] | 0 | 6.00 |
| RePAR 禁用图执行 | 27/125 | 21.6% | [15.3, 29.6] | 0 | 6.00 |

## 配对比较

| RePAR 对照 | 准确率差（百分点） | 配对 bootstrap 95% CI | 胜/负题数 | 双侧精确 McNemar p |
|---|---:|---:|---:|---:|
| Qwen3.5 0.8B 直接回答 | -4.0 | [-13.6, 5.6] | 16/21 | 0.5114 |
| Qwen3.5 0.8B + BM25 RAG | -4.0 | [-12.8, 4.8] | 13/18 | 0.4731 |
| RePAR 禁用图执行 | +0.0 | [0.0, 0.0] | 0/0 | 1.0000 |

Bootstrap 使用 20,000 次同题配对重采样、seed=42。统计检验未作多重比较校正。置信区间跨 0 时不能认定稳定提升。

## 全上下文与截断

直接回答中 **111** 题完整装入上下文；**14** 题超出窗口，按官方实现保留 prompt 首尾各一半。输入预算 261,632 tokens，为聊天模板和 128-token 输出预留空间。
主表是官方风格的可用窗口基线，并非所有题都完整输入了原文。RePAR 建索引时始终处理完整原文。

| 方法 | 基线能完整输入的题 | 基线需要截断的题 |
|---|---:|---:|
| Qwen3.5 0.8B 直接回答 | 27/111 (24.3%) | 5/14 (35.7%) |
| Qwen3.5 0.8B + BM25 RAG | 26/111 (23.4%) | 6/14 (42.9%) |
| Qwen3.5 0.8B + RePAR | 25/111 (22.5%) | 2/14 (14.3%) |
| RePAR 禁用图执行 | 25/111 (22.5%) | 2/14 (14.3%) |

## 按长度拆分

| 分组 | N | 直接回答 | BM25 | RePAR | 无图 RePAR |
|---|---:|---:|---:|---:|---:|
| long | 23 | 30.4% | 34.8% | 13.0% | 13.0% |
| medium | 44 | 25.0% | 20.5% | 27.3% | 27.3% |
| short | 58 | 24.1% | 25.9% | 20.7% | 20.7% |

## 按难度拆分

| 分组 | N | 直接回答 | BM25 | RePAR | 无图 RePAR |
|---|---:|---:|---:|---:|---:|
| easy | 34 | 23.5% | 23.5% | 17.6% | 17.6% |
| hard | 91 | 26.4% | 26.4% | 23.1% | 23.1% |

## 按子领域拆分

| 分组 | N | 直接回答 | BM25 | RePAR | 无图 RePAR |
|---|---:|---:|---:|---:|---:|
| Academic | 50 | 24.0% | 26.0% | 24.0% | 24.0% |
| Financial | 15 | 13.3% | 20.0% | 20.0% | 20.0% |
| Governmental | 23 | 30.4% | 26.1% | 26.1% | 26.1% |
| Legal | 14 | 35.7% | 21.4% | 7.1% | 7.1% |
| Multi-news | 23 | 26.1% | 30.4% | 21.7% | 21.7% |

## 成本与运行情况

| 方法 | 平均输入 tokens | 平均输出 tokens | 平均服务计算秒数 | P95 服务计算秒数 | 平均实测墙钟秒数 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 0.8B 直接回答 | 102948 | 8 | 14.35 | 48.43 | 14.99 |
| Qwen3.5 0.8B + BM25 RAG | 9053 | 8 | 0.53 | 0.66 | 0.64 |
| Qwen3.5 0.8B + RePAR | 34237 | 981 | 5.02 | 6.20 | 5.33 |
| RePAR 禁用图执行 | 34236 | 981 | 5.02 | 6.20 | 0.15 |

服务计算时间为 API 返回 prompt_ms + predicted_ms 的各次调用之和；墙钟时间包含排队、网络、API 请求缓存读取和本地工具执行。调用缓存命中也按原始实际 API token 和服务时间计入；因此复用已有相同请求不等于零推理成本。
完整索引共 125 个唯一上下文，45,783 个叶片、67,057 个树节点、700,754 条图边，累计首次构建 163.63 秒；该离线成本单独列出。

```json
{
  "repar": {
    "tool_counts": {
      "tree_search": 209,
      "read": 1
    },
    "mean_rounds": 5.0,
    "round_distribution": {
      "5": 125
    },
    "fallbacks": 3,
    "termination": {
      "budget": 125
    },
    "graph_used_samples": 0,
    "policy_revisions": 16,
    "tool_errors": 415,
    "duplicate_call_errors": 396,
    "finish_with_calls": 625,
    "two_retrieval_rounds_samples": 67,
    "json_errors": 0
  },
  "repar_tree": {
    "tool_counts": {
      "tree_search": 209,
      "read": 1
    },
    "mean_rounds": 5.0,
    "round_distribution": {
      "5": 125
    },
    "fallbacks": 3,
    "termination": {
      "budget": 125
    },
    "graph_used_samples": 0,
    "policy_revisions": 16,
    "tool_errors": 415,
    "duplicate_call_errors": 396,
    "finish_with_calls": 625,
    "two_retrieval_rounds_samples": 67,
    "json_errors": 0
  }
}
```

## 实现范围与解释边界

树采用保持原文顺序的连续区间分割，在近似均衡的候选边界中优先切断 TF-IDF 邻接相似度较低的位置；叶片约 240 words，优先保留段落/句子边界。TF-IDF 是稀疏词汇语义代理，不是训练过的神经语义编码器。
知识节点保留片段中显著的三个原文命题、概念词及因果/对比/时间线索；图边为共享实体、共享概念、词汇关联、原文相邻。它是一种带出处的软知识图，边本身不证明因果、蕴含或矛盾；最终判断依靠原文。
树查询执行层次 beam 路由，hybrid 模式允许两个倒排检索候选补充召回；图查询允许种子、跳数、关系过滤和返回数量。禁用图执行的消融保留相同控制器、工具描述和预算，在 dispatcher 处阻断图调用并将错误反馈给控制器。首次图访问以前的状态完全一致，可复用相同请求缓存；未尝试图访问的题两组轨迹可完全一致。这衡量图访问能力被阻断的影响，不是重新设计无图控制器的成绩。
控制器以 JSON 工具协议输出参数化调用，本地 dispatcher 执行；执行结果、反思、策略更新和终止原因逐轮保存。这不是伪造的固定查询轨迹，也不依赖服务端原生 function-calling 解析。
125 题上的一次运行不能证明通用提升，四位量化、tokenizer 和 2026 年的底座也与 2024 年论文设置不同。论文的历史结果只作同任务参考；不能把框架增益与底座升级的贡献混在一起。

## 复现与原始材料

```powershell
conda activate RePAR
python -m pip install -e .
python -m pytest -q
python scripts/prepare_data.py
python scripts/run_suite.py --config configs/qwen08_mtp.json --output artifacts\experiment_mtp
```

逐题对照：`artifacts\experiment_mtp/per_question.csv`；原始预测：同目录四个 `.jsonl`；完整 API 请求/响应：`artifacts/cache/llm`；每题 ReAct 轨迹：`artifacts/traces_qwen08_mtp`；汇总机器数据：`artifacts\experiment_mtp/summary.json`。
论文：https://arxiv.org/html/2412.15204v1#S4.T3；官方评测：https://github.com/THUDM/LongBench/blob/main/pred.py；数据：https://www.modelscope.cn/datasets/ZhipuAI/LongBench-v2。
