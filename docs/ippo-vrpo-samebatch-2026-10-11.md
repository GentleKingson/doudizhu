# IPPO 与 VRPO + CentralCritic 同 batch 性能对照

2026-10-11（香港时间），LocalServer / RTX 5070。相同 batch 下，IPPO 为 **108.85 条轨迹/秒**，VRPO 为 **68.46 条轨迹/秒**；VRPO 吞吐低 **37.11%**，每轮耗时长 **59.01%**。新增 CentralCritic 训练阶段是主要差距来源，GAE、Muon 和 Replay 不是首要瓶颈。

## 测量范围

- 完整斗地主、v1.3、Muon + AdamW、FP32 权重/梯度/状态、TF32、Inductor、seed=0、OMP=6，单 GPU/单进程。
- `rollout=256, collect=128, mini_batches=2`，训练 mini-batch 为 64。Policy 各 4 epochs；VRPO 另保留 Critic 4 epochs、CUDA Replay scale=16。
- 学习率 `4e-4`、clip `0.02`、regularization `0.1`，稳定阶段均 500 iterations；没有改变模型、GAE lambda 或更新顺序。
- 六个独立进程按 IPPO1 / VRPO1 / VRPO2 / IPPO2 / IPPO3 / VRPO3 交错运行。各 20 轮编译预热、30 轮 profiler-off 计时；每种算法合计测量 90 轮、11,520 条交付轨迹。吞吐为累计交付轨迹 / 累计 wall 秒数，再取三次运行的中位数。
- 两份独立 CPU/CUDA trace，各 20 轮编译预热，之后 wait=1 / warmup=1 / active=1。Trace 的 wall 不参与吞吐对照。
- 另各 20 轮预热、6 轮无 profiler 的互斥阶段计时，在阶段边界同步 CUDA，并细分训练前向、GAE/loss、反向和优化器。同步会改变异步重叠，因此这组数据仅用于归因。
- 短跑关闭 checkpoint、评测、参数统计和 W&B。使用相同种子和配置，但两种算法学到的策略不同，采样牌局/轨迹并不逐条相同；本次只比较计算性能。

环境：Python 3.12.15、PyTorch 2.14.1+cu130、CUDA 13.0、驱动 595.91.07、RTX 5070 12227 MiB。

## 不开 profiler 的基线

| 算法 | 运行 1 | 运行 2 | 运行 3 | 吞吐中位数 | 每轮平均耗时的中位数 | 测量期峰值已分配显存 |
|---|---:|---:|---:|---:|---:|---:|
| IPPO | 108.850 | 109.283 | 108.545 | **108.850 条/秒** | **1175.926 ms** | 5905.69 MiB / 5.77 GiB |
| VRPO + CentralCritic | 68.292 | 68.456 | 68.458 | **68.456 条/秒** | **1869.824 ms** | 6779.39 MiB / 6.62 GiB |

显存来自 `torch.cuda.max_memory_allocated()`，不包含 CUDA context 和 allocator 的空闲预留。5 秒一次的 `nvidia-smi` 原始记录也已保留；六次基线和两份 profile 的整个运行窗口中，驱动观测峰值为 8228 MiB，包含编译预热及 profiling。

IPPO 测量窗口内始终 25 个 compiled unique graphs，VRPO 始终 39 个，均与预热结束相同。所有训练指标均为有限值。两种算法的测量期平均轨迹长度分别为 85.90 和 84.32 个环境步，因此还保留真实完成游戏数、交付轨迹数和实际出牌步数三种计数，避免把 padding 或 pending buffer 当成真实样本。

## 阶段归因

下表是独立的同步诊断，每种算法 6 轮的均值，不是 profiler-off 基线的直接拆分：

| 互斥阶段 | IPPO ms/轮 | VRPO ms/轮 | VRPO 诊断 wall 占比 |
|---|---:|---:|---:|
| 采样及 rollout 指标 | 565.75 | 560.94 | 28.11% |
| 初始 bootstrap | 71.43 | 43.61 | 2.19% |
| 策略更新 | 574.33 | 620.83 | 31.12% |
| Critic 更新（含当前策略概率重算） | — | **767.74** | **38.48%** |
| Replay 写入 | — | 0.40 | 0.02% |

其他调度、EMA、日志和诊断导出均单独记录。同步诊断的总 wall 为 IPPO 1218.19 ms、VRPO 1995.23 ms，比各自基线中位数高 3.59% / 6.71%；不要把它们当成新的吞吐基线。

VRPO critic 阶段中，**当前 actor 前向 211.64 ms、critic 前向 161.00 ms、critic 反向 291.11 ms**，合计 **663.74 ms**，约为该阶段的 86.45%。这三项才是主要额外工作。IPPO 的 value head 与 actor 共享主干，参数总量 1,915,906；VRPO actor 为 1,915,649，另有独立 critic 5,456,897 个参数。CentralCritic 编码三名玩家信息并输出动作 Q 值，不能按“多一个小 value head”估计成本。

独立 trace 支持同一判断：VRPO critic 阶段有 713.47 ms CUDA kernel 工作，占全轮 kernel 时间 42.32%；其中 actor 前向 210.94 ms、critic 前向 159.96 ms、critic 反向 288.72 ms。双方策略网络的 CUDA 前向/反向时间基本相同：IPPO 205.39 / 328.75 ms，VRPO 205.21 / 328.05 ms。两份 trace 的采样轮数和真实步数不同，阶段差值不能当作逐条相同轨迹的因果估计。

GAE/loss 的重复计算确实有成本，但属于第二层优化：VRPO Policy 为 55.31 ms，Critic 为 52.92 ms，合计 **108.22 ms/轮、诊断 wall 的 5.42%**。两个 scope 在 trace 中发射了 26,672 个 kernel，而 kernel 工作仅 27.55 ms，存在小算子启动/调度开销。即使理想地消除整个 GAE/loss 阶段，按这份诊断估算也只有约 **5.7%** 的吞吐提升上限；实际融合只能消除其中部分开销。

Actor/Critic 优化器合计 61.31 ms，约 3.07%；Replay 写入与抽样合计约 1.51 ms。当前 Muon 已经按形状批处理 Newton–Schulz 矩阵运算，不能把尚未实现的“Muon batching”当作新提速点。

CPU annotation 可能包含等待前面已排队的 GPU 工作。例如 trace 中 critic metrics 的 CPU inclusive 时间为 64.90 ms，但同步细分计时仅 0.10 ms。因此不能把 CPU scope 时间直接解释为日志计算很慢，也不能把 CPU、GPU、嵌套 sampler 时间相加。Kernel 之间的间隔还可能包含 copy/memset，不能全部当成 CPU launch 导致的 GPU 空闲。

## 尽量保留训练质量的提速顺序

1. **先试网络算子 autotune。** 保留当前网络、FP32/TF32、mini-batch、4/4 epochs、Replay 和优化器设置，仅对已编译的 Agent/Critic 试 `mode="max-autotune-no-cudagraphs"`。这是改动最小、直接针对主要网络计算的候选。先检查输出、loss、裁剪前梯度、clip 分支及更新后参数，再做三组独立完整迭代 wall 对照。它会增加编译时间，收益尚未实测。[PyTorch torch.compile modes](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)

2. **随后减少无效尾部 padding 的网络计算。** 实际固定 horizon 为 177，测量轨迹平均仅约 84–86 步。先补充每个现有随机 mini-batch 的最大有效长度分布，再从少量长度桶开始，例如只试 128 / 177。只裁剪桶后全部无效的时间步，保留原 mini-batch 成员、顺序、全部真实出牌及三名玩家信息；避免为了长度分桶重排样本，改变 batch 内优势归一化。保持 mask、位置编码、GAE targets、loss reduction 的语义，验证完整有效部分的输出与梯度。模型大小、每个样本的训练次数和 Replay 不变。平均 padding 比例不能直接当成可实现加速比，因为裁剪长度受 batch 最大值约束；还需逐 frame 检查编译缓存和重编译上限，避免新增形状导致回退到 eager。

3. **再融合 FP32 Q-GAE / loss。** 对纯张量计算尝试 `torch.compile`，必要时再写融合 kernel；继续在每个 Policy/Critic mini-batch 计算当前策略概率、Q(lambda)、TD1 和优势。保留 FP32 累积、gamma/lambda、mask、PPO clip、正则和归一化。不要把 8+8 次计算改成每轮一次，也不要从旧 Replay 缓存中复用过时的策略概率。这项应按几百分点的候选收益衡量，不应承诺接近 IPPO 的速度。

Actor 在 critic 阶段不更新，可探索把同一组输入的 actor 概率计算合并成更大的无梯度 batch，再按原 critic mini-batch 索引取回。仅允许复用同一 actor 版本、同一输入的结果；跨迭代缓存旧策略概率会改变目标。此方向还需检查峰值显存和训练模式下的随机层，当前没有实测收益。

暂不优先减少 Critic epochs、缩小 critic、减少 Replay、降低精度或更换优化器。这些会改变估计质量/训练行为，属于后续消融。若前三项效果有限，再测试 `collect=160, mini_batches=2`（训练 mini-batch=80），仍保留 4/4 epochs；它会改变梯度统计及样本与 schedule 的关系，不能归类为保持算法结果的优化。

每项分别通过数值/梯度检查和至少三组 profiler-off wall 比较后再合并；正式训练前还需用相同样本预算、多个种子和固定牌局评测胜率/收益。当前数据不能给出优化后吞吐或长期强度保证，也不能证明 VRPO 的额外网络工作能无损地全部消除。

## 检查与复现材料

IPPO 复用了现有 VRPO 诊断器，补齐相应阶段。移除诊断上下文后，IPPO 更新循环 AST 与原版本一致；CPU 训练输出/计数一致性等检查 7 项通过，真实 CUDA trace 检查 2 项通过。两份正式 trace 均 `PROFILE_VALID`，没有编译事件、缺失阶段或未关联的 CUDA kernel。

服务器在独立目录运行源码快照，未修改 `/root/doudizhu` 的代码。服务器基线 git SHA 为 `acdcf199cf422e2b683cb22142b8e7cb33b14460`；实际训练源码为本地 `6f3070686f4e62cb128cc28fcc8886fd2d0edac2` 加本次诊断补丁，源码包 SHA256 为 `5134ad1f008d982542a0ff6cb01aaecc1a9f1f5dc84ad3fb1d19c849afc7175b`。八个主要运行及两次阶段计时的逐文件源码哈希相同，不能把快照中继承的服务器 git SHA 误认为完整运行源码版本。

IPPO 同步诊断跑满 26 轮后，最初 2 ms 的阶段闭合检查因为一次约 31.74 ms 的 CSV 导出/调度差额而失败；全量阶段、细分调用次数、batch 和编译图数已重新核验，复用了这份完整数据。该差额单独计入诊断导出开销，6 轮平均 5.64 ms，未计入任何算法阶段。恢复过程中还保留了一次输出文件名冲突和一次差额阈值检查失败；它们没有启动额外训练。最终 VRPO 阶段计时正常完成，原始失败日志均保留。

成功的主流水线及阶段复核 systemd 单元均已退出，GPU 回到 0% / 18 MiB。服务器原始目录为 `/root/doudizhu/logs/samebatch-profile-20261010-170247`。

本地材料位于 `logs/samebatch-profile-2026-10-11/`：

- [汇总、原始文件 SHA256 与计算结果](../logs/samebatch-profile-2026-10-11/summary.json)
- [实际执行的分析脚本](../logs/samebatch-profile-2026-10-11/analyze.py)、[配置协议](../logs/samebatch-profile-2026-10-11/protocol.json)、[运行与下载校验记录](../logs/samebatch-profile-2026-10-11/receipt.json)
- `results/` 包含六次 baseline、两份原始 trace/CPU/CUDA 算子表/阶段表、两次同步阶段计时、训练指标、5 秒显存记录、完整配置/源码哈希/命令、状态及失败恢复日志。
- 服务器与本地压缩档案 SHA256 均为 `9acb1c17c65eef6298c2a6b64b0af4de3af3296e924c90be53501330ae38adb4`。

没有执行新的 1 小时训练、实战评测或优化候选实验；以上提速方案的实际收益和长期训练质量仍待验证。
