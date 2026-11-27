# 数据与资源发布范围

更新时间：2026-11-28 01:05，Asia/Shanghai。

## LongBench v2

[`longbench_v2.json`](longbench_v2.json) 是完整的 **503 题**固定快照，大小 **465,490,535 bytes**，通过 Git LFS 发布。本项目的 LongBench 评测只使用其中 **125 题 Multi-Document QA**，不是 503 题全榜。

- 上游数据：[THUDM/LongBench-v2](https://huggingface.co/datasets/THUDM/LongBench-v2)。数据卡副本见 [dataset_readme.md](../third_party/dataset_readme.md)，其中声明 `license: apache-2.0`；许可文本见 [LICENSE.LongBench-v2](LICENSE.LongBench-v2)。
- 实际下载源：[ModelScope ZhipuAI/LongBench-v2](https://www.modelscope.cn/datasets/ZhipuAI/LongBench-v2)，revision `6a226395b12a1438f40dfabbe0c5ea55f6a102bc`，原文件 `data.json`；本地仅改名，未改内容。
- SHA256：`15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`。
- [无答案的子集清单](../artifacts/dataset_manifest.json)保存题目 ID、原文哈希、领域与长度；索引构建仍不接触问题、选项或答案。
- [LongBench 上游代码许可](../third_party/LICENSE.LongBench)为 MIT；这是代码许可，不应混作数据的 Apache-2.0 许可。

在仓库根目录恢复与验证数据：

```powershell
git lfs install
git lfs pull --include='data/longbench_v2.json'
# 不通过 LFS 下载时，以下命令下载并校验同一快照
python scripts/download_assets.py --dataset-only
```

原始数据与历史实验归档不同：预测、冻结索引、请求和源码快照不包含在这个 JSON 中，不能仅凭下载数据就声称通过了历史实验审计。

## MATH 2025 全国Ⅰ卷

本地评测使用 19 题、150 分的规范化题目与参考解答，以及连续教材语料。核对来源、字段和修正见 [MATH 数据说明](MATH_2025_NATIONAL_I/README.md)。目前没有确认原试卷/解析 PDF、规范化参考解答和人教版教材全文的再分发许可，因此这些正文没有上传。该目录的历史说明记录完整本地资源，不代表所列 PDF/JSONL 已随仓库分发。

## 其他资源核对

| 资源 | 发布情况 |
|---|---|
| 项目代码、评测适配、Web 静态资源、配置、文档 | 普通 Git 文件 |
| LongBench v2 完整 JSON | Git LFS；503 题、465,490,535 bytes |
| 模型来源清单、0.8B 标准/MTP 下载清单 | 已纳入 Git，可追溯固定版本和 SHA256 |
| 最终 v5 成绩/成本汇总、冻结协议及精选审计摘要 | 已纳入 Git；原始运行不包含在摘要中 |
| 模型权重约 8.00 GB、运行产物约 56.11 GB | 本地资源；不自动上传 |
| 生成的 `webserver/data/` 约 3.39 GB、用户知识库与会话 | 本地缓存和用户数据；缺失时 Web 显示空状态 |
| 论文全文、未确认再分发许可的 CoT-RAG/T3 上游副本 | 不发布，不能宣称是已可复用的 baseline |

大小来自 2026-11-28 发布核对，按十进制 GB 近似；这些文件数/容量不是运行时依赖要求。MATH 原文与模型等外部资源应按原来源和许可自行准备，保持配置声明的版本及路径。
