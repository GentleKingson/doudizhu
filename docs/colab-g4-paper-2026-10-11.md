# G4 论文训练参数对齐与吞吐复测

状态：已完成；实测结果已审计，G4已释放。旧版吞吐数字不适用于本版。

## 固定条件

官方仓库 https://github.com/AtaraxosAI/doudizhu ，提交 `acdcf199cf422e2b683cb22142b8e7cb33b14460`。独立干净克隆，调用原始 VRPOTrainer；本地其他未提交训练修改未参与测试。

训练参数参照 [Nature S7/Table S27](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf) 与 [VRPO Appendix E](https://arxiv.org/html/2605.19235v1#A5)：rollout512、collect2048、mini_batches4、replay64、EMA0.999、lr_stable5000、reg_stable500；actor/critic各4 epochs，Muon/AdamW、4e-4学习率、λ0.95、γ1、clip0.02、正则0.1、裁剪1、原模型与FP32/TF32不变。

官方完整replay固定布局为487112 bytes/轨迹，64×2048占59.46 GiB。使用已有CPU replay选项以保留完整容量，拒绝通过减少collect/replay/epochs、改变minibatch大小或降低精度换取速度。CPU存储不改变字段值或采样分布，但随机数发生器设备不同，不保证逐位一致。只搜索CPU线程4、6、8；保留论文rollout512。

## 测量与约束

每组独立进程、共享编译缓存；初筛16轮热身+8轮测量仅用于选候选，replay尚未填满。候选以seed0和seed1分别运行64轮热身+16轮测量，8轮一个计时块。检查第64轮起replay完整达到131072条，训练指标有限，测量阶段无新增编译。

显存每200ms采样，保守峰值取max(NVML峰值, PyTorch reserved峰值+512MiB)，要求显存余量≥15%，主机可用内存≥20%。独立监控主机RAM，低于10%中止。无训练checkpoint，避免不必要I/O及额度。吞吐为总轨迹数/测量总秒数，不是各组速率简单平均。

本次目的为保持论文训练配方后进行工程性能调优；不将短程吞吐、有限loss或无OOM当作长期牌力证明。未加入belief训练与在线搜索，非完整Ataraxos复现。

## Notebook兼容性

保留Google Drive目录读取、最新/指定checkpoint续训，默认训练120分钟、每30分钟保存、保留最近10份，均保持现有控制。额外遵守论文40000轮总上限。旧checkpoint可继续加载，但无法追溯修正此前训练历史。原checkpoint不包含replay/RNG/环境状态，续训需重新积累64轮replay。

保存仍先本地落盘、Drive临时文件复制、原子发布后滚动清理；仅清理当前运行目录匹配ckpt_数字.pt的文件。训练结束保存，Drive同步成功后释放，同步失败保留实例便于恢复。

实际Drive交互授权及端到端写入未执行；离线验证保存失败保护、滚动保留、周期保存与结束、续训格式、释放顺序和论文参数传递。

Notebook通过nbformat格式检查；HTML预览已生成，但内置浏览器安全策略不允许file://页面，未完成视觉检查，也未尝试绕过该限制。

## G4 实测结果（论文参数约束，2026-10-11）

选定CPU线程 **8**。完整replay双种子复测的合并吞吐为 **175.659轨迹/秒**、12919.2环境步/秒、8554.0实际决策步/秒。

全程NVML显存峰值 **52.274 GiB**，保守显存余量 **43.318 GiB**；最低可用主机内存 **111.403 GiB**。硬件为RTX PRO6000 Blackwell Server Edition，PyTorch2.11.0+cu130。

3组线程初筛（16热身+8计量）；选定配置以seed0/1各运行64热身+16计量，确保replay已填满131072条并经过覆盖写入。所有合格区间无新增编译，训练指标有限。计量不含编译、保存和评测，不能代表端到端含Drive保存的速度。

线程间小差异不证明统计显著优势；这是有限候选内选择，不是全局最优或牌力证明。旧版175.304轨迹/秒使用不同学习参数，不作同条件比较。

实测训练未保存checkpoint。G4已关闭。GPU作业共51.01分钟；账户余额读数差7.12单位，不是精确逐作业账单。

| 阶段 | CPU线程 | 轨迹/秒 | 显存峰值GiB | 保守余量GiB |
|---|---:|---:|---:|---:|
| screen-00 | 6 | 163.354 | 52.280 | 43.312 |
| screen-01 | 4 | 163.246 | 52.216 | 43.377 |
| screen-02 | 8 | 164.043 | 52.274 | 43.318 |
| verify-0-seed0 | 8 | 175.400 | 52.274 | 43.318 |
| verify-0-seed1 | 8 | 175.919 | 51.981 | 43.611 |

交付：[notebook](../notebooks/colab_g4_vrpo_centralcritic.ipynb)。原始记录：[results.json](../logs/colab-g4-paper-2026-10-11/results.json)、[audit.json](../logs/colab-g4-paper-2026-10-11/audit.json)、[释放回执](../logs/colab-g4-paper-2026-10-11/receipt.json)。

最终校验：9项离线测试、Notebook格式及全部代码单元语法通过；远端186个受版本控制文件的SHA256均与固定官方提交一致；5组原始记录已独立复算，训练checkpoint为0。Drive真实授权/读写和完整Notebook端到端执行未完成，长期牌力未测试。
