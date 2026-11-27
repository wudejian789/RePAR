# 三种Baseline全量评估协议与运行入口

用户已授权RAPTOR、HippoRAG2、HyperGraphRAG在既有LongBench v2 Multi-Document QA **全部125题**上运行，合计375项；不是503题全领域评测。取消上一轮按长度筛选及180秒/题门槛，不因某题慢而跳过余下题。最终结果见 [结果页](BASELINE_FULL_RESULTS.md)，状态以实际progress与进程为准。

## 底座与并行执行

仍使用本机Qwen3.5-0.8B标准Q4_K_M、llama.cpp b12500，关闭thinking和MTP，temperature=0.1、seed=42。没有换模型、改prompt或根据pilot答案调参数。9091及数学补测保留；新增9092/9093两个相同权重服务，每服务8槽位、总ctx=524288、每槽65536。全文仍全部进入树/图建库，生成只处理分段或证据；窗口配置单列部署身份，不混用旧缓存。

独立合成请求按串行→16并发→16并发→串行测试，平均输出吞吐约234→1246 tokens/s，约5.33倍。探针在既有数学补测共享GPU时执行，不是LongBench加速倍数；收益来自双副本与连续批处理组合。实测常驻显存约22.6GB/24GB。服务props、启动参数、探针请求和计时在artifacts/deployment。9091没有被终止或改配置。

三个方法各一个题目流并行；Hippo/Hyper每题最多8个独立抽取调用，中央loopback gateway全局限制各服务8个在途请求。Hippo保留每片段NER→triple语义；Hyper用线程执行同步HTTP，保持每片段实际返回后的历史/gleaning顺序。RAPTOR保留作者串行摘要，未改变聚类或节点编号。

## 冻结与兼容

配置见 [baseline_full.json](../configs/baseline_full.json)，输出为artifacts/baseline_full_qwen08_dual_v1。protocol.json冻结125个唯一ID、原文/数据SHA、模型配置、服务props/权重SHA、encoder、方法参数和源码。执行使用该目录source快照，工作区之后的文档/报告工具更新不改变正在运行的算法。

真实请求按完整body和新deployment身份缓存，使用本轮新目录，不导入pilot缓存。断点只复用真实请求完全一致的缓存并计数；HTTP错误、原始输入输出、重试、backend、排队和服务时间留存。最多两次transport尝试只处理网络/5xx，不重写prompt或按答案重生成。单backend请求600秒是故障防护，不设题目总耗时上限。

保持上一轮MiniLM CPU encoder、240词无损分段、token caps与检索方式。普通接口默认workers=1，本实验显式workers=8。并行可能影响浮点采样和上游插入顺序，使用新版本与新缓存身份。新增客户端保存线程独立last_call，避免finish_reason/usage串线；索引身份包含客户端源码hash和worker参数。

## 评分与完成标准

标准答案只在作答后评分。最终reader使用已有官方RAG模板、128-token预算与答案grammar。无证据、解析失败或作者算法抽取失败原样保留，计入125分母，不偷偷换成BM25。Hippo官方dense fallback单列，不冒称PPR。基础设施崩溃会阻止完成报告，不能用空结果伪造模型错误。

每方法全部125唯一ID才生成正式summary/REPORT；失败计0并单列。成本区分建库、encoder、在线检索/作答、排队、真实服务时间、缓存和物理传输。三个条件并行，逐题时长之和不等于整个实验墙钟，不能与旧串行结果直接比较加速。

finalizer等待全量summary后审计375项、原文/代码/模型身份、每个完成响应对应真实backend请求、grammar及分数，生成结果页、Wilson 95%区间、错误分布和历史RePAR v2配对精确检验。历史RePAR仅复用既有125题预测并注明hash，不冒称重测。题目独立同分布假设未被证明，统计区间仅作描述性参考。

先前3道pilot题已被观察，未据其标签调prompt/参数，不能称为完全未见测试确认。相同最终LLM不等于全部资源相同：encoder、固定示范、离线生成和不同证据预算继续披露；本次不是论文默认强模型精确复现。

## 运维入口

```powershell
python scripts/baseline_full_progress.py
python scripts/run_baselines_full.py --resume artifacts/baseline_full_qwen08_dual_v1/protocol.json
python scripts/finalize_baseline_full.py --run artifacts/baseline_full_qwen08_dual_v1 --historical artifacts/experiment_v2/repar.jsonl --watch
```

仅在原coordinator退出后resume，文件锁防止重复调度。完成final按ID跳过，中断attempt另开目录，仅准确请求身份缓存可复用。后台启动不代表完成：375项覆盖与最终审计passed才是交付条件。watcher只读预测并汇总，不产生模型请求或重跑benchmark。
