# LocalServer 可用性与 S1a/K1/E1/G1 最小 GPU 预算

2026-10-11，香港时间。本次只读 SSH、驱动/软件查询及历史耗时分析；未启动 CUDA kernel 测试、训练、部署或新实验。以下预算是提案，尚未执行。

## 可用性

10:38:06–10:38:39 实测：LocalServer 可通过既有 SSH 配置连接。GPU 0 为 RTX 5070（UUID `GPU-46dce763-982c-cb97-2004-b7770ccdd7eb`），总显存 12,227 MiB，已用 18–20 MiB，利用率 0%，温度 35°C，空闲 P8，功率约 11 W、上限 250 W；compute-apps 查询为空。10:41:11 未发现名称含 VRPO/MARVEL/doudizhu/GPU 的运行中 systemd 服务，旧实验状态为 completed。查询时 CPU 列表中的高占用 Python 是本次依赖探测进程，随后已退出。

服务器环境：Python 3.12.15、PyTorch 2.14.1+cu130、CUDA runtime 13.0、驱动 595.91.07；`torch.cuda.is_available()` 为 true，识别 1 张卡，构建包含 sm_120。pytest、numpy、nanopokerlib 均可导入。根盘约 1.7 TiB 可用，/tmp 约 15 GiB 可用。结论为**当前空闲且软件可识别，可安排下一阶段**，未做负载健康测试，也未预留设备。

远端仓库 clean，HEAD 为 `acdcf199cf422e2b683cb22142b8e7cb33b14460`，不是本地 B0。不要直接在该 checkout 跑当前候选。已有 `/root/doudizhu/logs/vrpo-stable-speed-20261010-1826` 保留 source bundle、约 98 MB frozen_batch、numeric.py、default/sequence.pt 和 reusable GAE；source bundle/numeric/原 GAE 哈希与本地档案匹配。两个提交间 nanopokerlib 源码没有差异。后续在新隔离实验目录重建本次 L0 和独立补丁，保留远端主工作区及旧档案。

## 最小预算建议

建议先分配 **1 张 RTX 5070 × 30 分钟占用 wall 硬上限（0.5 GPU·h）**。这是一轮可提前停止的正确性验证与性能粗筛；不承诺 30 分钟内一定完成全部冷编译或得到性能通过结论。CPU 补工具、准备源码/合同和核验档案在占用 GPU 前完成。

| 顺序 | 工作与最多运行数 | 首轮分配 |
|---|---|---:|
| 1 | S1a 固定真实输入 baseline/repeat 两进程；G1 单独两轮候选一进程；K1/E1 实际采样两轮 baseline/repeat/K1/E1 四进程；一个 CUDA 专属测试进程 | 8 分钟 |
| 2 | 正常连续块 A/A pilot，一对、两个独立进程 | 5 分钟 |
| 3 | K1 → E1 → G1，各自一对独立 A/B，共六个性能进程；任何未过数值门的项不跑 | 14 分钟 |
| 4 | 有界结束、记录失败/超时、状态与产物收尾 | 3 分钟 |
| 合计 | 最多 8 个数值/测试进程 + 8 个性能进程，全部串行 | **30 分钟** |

阶段额度是硬预算分配，不是测得的耗时上界；只有阶段提前结束的剩余额度可留给后续，不能增加运行次数或超出总额。到时保留证据并标 `INCONCLUSIVE`，不自动重试。若数值阶段 8 分钟不足，首轮可止于数值覆盖，性能尚未运行的项如实记录。

若首轮有继续价值，可另决定增加 **30 分钟，总上限 60 分钟（1 GPU·h）**：给每个保留候选增加两对 A/B，全部三项继续时总计 20 个正常模式性能进程（初始 A/A 两个 + 三候选各六个）。这仍是三对初筛，不是保证确认 0.5% 收益所需的统计样本数；最终确认数量取决于噪声，另行在新数据前冻结，不能无限加跑到显著。

首轮各候选仅一对，报告原始 wall 与点估计，不伪造配对方差或置信区间。A/A pilot 一对只查明显不稳定和观察器问题，不足以估计可靠统计功效。三对方案若采用配对 log-wall 比值的 Student-t 区间，应事先冻结该探索性口径并明确 df=2；三对不能在每候选内精确 1:1 平衡 AB/BA，使用交错顺序披露。最终确认应采用偶数对和独立数据，不从多个候选中只展示最佳一次。

## 预算依据

原 [timing.json](../logs/vrpo-stable-speed-2026-10-11/results/default/timing.json) 记录 strict baseline 35.069 秒、[repeat](../logs/vrpo-stable-speed-2026-10-11/results/repeat/timing.json) 15.791 秒、[G1 candidate](../logs/vrpo-stable-speed-2026-10-11/results/candidate/timing.json) 35.769 秒，合计 86.629 秒。六个 50 轮性能进程为 136.379 / 134.905 / 103.633 / 104.535 / 104.251 / 103.783 秒。后四均值 104.051 秒，前两均值 135.642 秒，额外量约 31.591 秒。

首次正常模式进程之前已经跑过 strict 工具，所以这些数字**不是完全空编译缓存的上界**。旧约 56 秒只含 30 轮测量，不能用于整个实验预算。新连续块计时需要自己的 A 基线，不把旧逐轮同步 CSV 当分母。

用于预算、并非性能预测的估式：`正常进程数×104.051秒 + 四模式各31.591秒 + strict三进程86.629秒 + 实际采样/专属检查预留300秒`。首轮八个性能进程约 22.4 分钟；全部按首次正常进程耗时算约 24.5 分钟。三对方案约 43.2 分钟，全部按首次正常进程耗时算约 51.7 分钟。30/60 分钟上限用于吸收一部分不确定性；新编译或新验证代码的耗时仍未测。

## CPU 先补 S1a，再使用 GPU

- 复用原 numeric/audit，不改训练器；补实际 PG 分支/系数与 autograd 对照，保留原 ratio clip 指标；加入 Python/NumPy RNG 起止恢复与比较、逐参数 shape/dtype、零/单侧零/极小梯度自检，以及每次输入内容摘要/索引/更新计数。实际 PPO 边界仍用已有真实冻结数据。
- 正常斗地主的有效玩家只可能为 -2/0/1/2：[初始状态](../nanopokerlib/cpp/games/doudizhu/core/doudizhu_core_state.hpp) 设 environment chance=-2，[发牌结束/玩家轮转/终局重置](../nanopokerlib/cpp/games/doudizhu/core/doudizhu_core_step.hpp) 没有 -1 路径；[HistoryBatch](../nanopokerlib/cpp/core/types/history_batch.hpp) 的 -1 默认值是未写入 padding。**真实有效 -1 对本游戏记 N/A**，保留共享算法 -1/-2 合同单测，无需 GPU 寻找不可达状态。这是对上一轮“缺真实 -1 数据”判断的收窄，不放宽 loss mask 语义。
- 冻结两轮 A/A 测网络、GAE、梯度、更新后模型/优化器、输入/Replay/全部 RNG 身份。每进程两轮应有 16 次 Policy 和 16 次 Critic 更新；A/A 必须逐位一致。
- K1/E1 验证另走真实 Sampler 和环境，不能沿用 `collect→full.clone()` 的绕过路径。baseline/repeat/K1/E1 各自两轮，逐步核对转换张量、概率/动作、terminal mask、cache/rebuild、完整 rollout、Replay、参数/优化器及 RNG；确认候选调用次数非零。K1 补 CUDA 空/稀疏/密集/全量 reset、别名、未选中行与后续输出；所有搬运结果逐位一致。
- G1 只补 11 次不同输入/key、旧 target 生命周期、8 图上限及第九个 key eager 回退的 CUDA 测试，再用已有冻结两轮工具验证当前补丁；不重做已完成的 GAE 微基准。

## 运行合同与停止条件

所有 A/B 从同一个 L0+S1 计时补丁重建，K1/E1 两侧 GAE 均关闭；G1 比较仅 Graph 候选不同，K1/E1 均不叠加。数值模式严格确定性；正常测速两侧均采用原生产开关，不混入 strict 选项。固定 FP32/TF32、默认网络编译、Muon/AdamW、4/4 epochs、rollout=256、collect=128、2 minibatches、Replay scale=16、6 CPU 线程。每个性能进程 20 轮热身 + 30 轮连续块测量；编译不稳定则本次结果作废，不现场改变测量长度追求通过。

性能候选顺序 K1、E1、G1 只表示优先级，互相独立。完整 reset 路径包含 `nonzero` 同步成本。每项分别保存完成游戏/交付轨迹/环境步/commit 步/更新数及分布；相同 seed 不等于相同工作量。首轮性能关闭评测/保存以对应既有短跑口径，不宣称已经覆盖正式评测/保存开销。

建议单进程显存预算 10 GiB、新档案磁盘预算 10 GiB；历史整进程约 7.5 GiB，当前卡约 11.94 GiB，但不是新形状显存保证。任何 A/A 不一致、非有限值/非法动作/身份或计数错误，停止相关候选；OOM、测量期编译、其他 GPU 进程干扰、资源/时间上限、覆盖不足，记录准确失败或 `INCONCLUSIVE`，不自动加跑。正式性能门仍为 wall 降幅点估计至少 0.5% 且95%配对区间下界>0，不能拿单对点估计放行。

本提案不含 A1/P1/C1、J1 联合版本、正式保存/评测集成、长期质量或 time-to-target；不重启 BF16/autotune/稀疏研究。确认 GPU 可用不等于已经批准实验，也不保证小收益一定能被统计确认。
