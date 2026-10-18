# VRPO：策略梯度误差放大定位与保留原运算的提速方案

找到能通过现有数值门槛的方案：两网络保持 default，仅使用 CUDA Graph 重放原样的无梯度 GAE 运算。固定输入的 34 个输出/loss/梯度检查、两轮更新后模型/优化器状态及最终 RNG 均逐位一致。三组真实自对弈短跑的吞吐中位数从 68.20 提升到 68.89 条训练轨迹/秒，本次观察提升 1.01%。相同输入的独立 GAE 计时从 6.12 ms 降至 1.43 ms。

## 实验范围与保持的条件

2026-10-11 02:25–02:47（香港时间），LocalServer RTX 5070，Python 3.12.15、PyTorch 2.14.1+cu130、CUDA 13.0。完整斗地主，v1.3，FP32/TF32、AMP 关闭、rollout=256、collect=128、2 个 minibatch、策略/critic 各 4 epochs、Replay scale=16（CUDA）、Muon+AdamW、base_lr=0.0004、seed=0、6 个 CPU 线程。模型大小、学习率、调度、训练源码、冻结牌局、初始权重和数值门槛与上一轮一致。

原有门槛仍为输出/loss atol=1e-5、rtol=1e-4；裁剪前梯度余弦 ≥0.99999、相对 L2 ≤0.1%；裁剪判断不一致比例 ≤0.1%。没有放宽门槛，也没有减少 epochs 或 Replay。

## 误差如何放大

重放上一轮“仅 CentralCritic autotune，coordinate descent 关闭”的 16 次真实策略更新。默认权重和优化器在第 8、16 步与原始记录逐位一致，重放有效。PPO 梯度系数公式也逐项对照了 torch.autograd.grad。

起点是 critic 的 Q 值舍入变化：首步最大差为 8.00937e-7，优势最大差为 7.15256e-7。策略首步输出相同，原始优势相对 L2 差异仅 4.79444e-8。随后两条优化器分支及 PPO 的梯度分支切换将扰动放大。

| 检查环节 | 实测证据 | 结论 |
|---|---|---|
| 优势归一化 | 16 个 minibatch 的优势标准差为 0.84885–1.25290；归一化前后误差相对 L2 比为 0.932–1.020 | 本批次没有标准差接近零的问题，归一化没有明显放大 |
| Muon | 第一更新的 Nesterov 输入差异 0.000432%，5 次 NS 后输出差异 0.069812%，约 161.6 倍 | 非线性矩阵迭代对这个微扰敏感，是早期放大环节 |
| AdamW | 第一更新的实际最大权重变化差为 2.71630e-5，发生在 state_mlp.0.weight；Muon 分组最大变化差为 7.15256e-7 | AdamW 分组也贡献早期扰动，实际分组需要核对 |
| PPO 裁剪 | 第 16 步 3813 个有效样本中 1 个实际 PG 梯度分支不同；PG 系数相对差 2.42665%，固定基线分支后仅 0.01043% | 少量边界跨越可以造成明显梯度跳变 |
| 全局梯度范数裁剪 | 16 步策略梯度范数为 0.333–0.795，均低于阈值 1.0；裁剪前后相对误差相同 | 本轮未触发该裁剪，不是这里的放大来源 |

### Muon 与 AdamW 的具体证据

第一步 Muon 的 NS 输出相对 L2 差异逐轮为：

| NS 次数 | 输出相对 L2 差异 |
|---:|---:|
| 1 | 0.003391% |
| 2 | 0.007992% |
| 3 | 0.017494% |
| 4 | 0.039064% |
| 5 | 0.069812% |

这些数值是固定真实梯度重放的结果，NS 第 5 轮也核对为逐位相同的原始 NS 输出。Muon 的范数归一化及后续矩阵迭代改变了扰动的相对大小；本次没有测矩阵奇异值谱，不能将放大进一步归因于具体某个奇异方向。

实际优化器分组中，state_mlp.0.weight 与 action_mlp.0.weight 归 AdamW。第一步 state_mlp.0.weight 的梯度相对差为 0.000145876%，权重更新相对差为 0.0234395%。AdamW 的逐元素自适应更新对接近零的梯度元素也可能敏感；最大绝对更新差是上述 2.71630e-5。这里同时报告实际分组和更新差，避免仅按参数是否为矩阵来判断其优化器。

### PPO 分支与日志统计的区别

实际有梯度的 PG 系数取决于归一化优势的符号及 ratio 是否跨过 0.98/1.02。日志 clip_frac 只统计 abs(ratio-1)>0.02，两者含义不同。第 8 步已出现 1 个实际分支差异，第 16 步也有 1 个；这两步优势符号未变化，差异与 ratio 的边界跨越有关。

第 16 步固定基线分支的 counterfactual 保留候选的优势和 ratio，只固定 active mask。该测试把 PG 系数误差从约 2.42665% 降到 0.0104286%，支持裁剪边界是后期明显放大环节。它不是对完整参数梯度的可加因果分解，也不包含 KL 正则与模型 Jacobian 的变化，不能解读为解释了某个比例的全部参数梯度误差。

上一轮实际策略总梯度最大相对 L2 差异为 1.15899%，尽管 32 次实际优化 loss 全部在容差内。小的平均 loss 差异不能代替逐步梯度检查。

## 通过验证的提速实现

两网络均使用 default，coordinate_descent_tuning 保持关闭。GAE 使用原 eager 运算捕获 CUDA Graph：按输入 shape、stride、dtype、device 和 gamma/lambda 缓存；首次在 side stream 热身 3 次后捕获；每次复制新输入再重放，并克隆输出以防下一次调用覆盖仍在用于 loss/backward 的目标。

只减少 GAE 的 CPU 发射开销，不改变归一化、PPO 裁剪、Muon/AdamW、精度、模型或训练目标。网络和优化器未纳入捕获。[PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)说明了静态输入、side stream 热身和输出存储的要求。

6 项实验检查、可复用入口的 2 项检查通过。两轮冻结更新中，34 个检查逐位一致，所有输出/loss/梯度最大差为 0，裁剪判断差异为 0；更新后模型、优化器状态和随机序列也逐位一致，原门槛全部通过。

## 吞吐与显存

每种模式 3 个独立进程，每次 20 轮热身、30 轮测量；每轮训练使用 128 条完整轨迹，测量工作量为每次 3840 条。顺序为 default-1、candidate-1、candidate-2、default-2、default-3、candidate-3。profiler 关闭，测量期网络编译计数均稳定在 39；候选每进程只捕获 1 次 GAE，50 轮共调用 800 次。

| 配对 | default 轨迹/s | GAE Graph 轨迹/s | 变化 |
|---:|---:|---:|---:|
| 1 | 68.0253 | 68.8924 | +1.275% |
| 2 | 68.3078 | 68.9739 | +0.975% |
| 3 | 68.2050 | 68.6957 | +0.720% |
| 各模式中位数 | 68.2050 | 68.8924 | +1.008% |

三次测量区间 PyTorch peak allocated 的最大值为 default 6799.06 MiB、候选 6786.16 MiB，基本持平。每 5 秒 nvidia-smi 记录到的全进程峰值分别为 7680、7674 MiB；它包含编译与热身，不能作为稳态显存节省结论。

这是相同 batch、参数与种子的真实自对弈测速，牌局和有效步数没有逐局锁定。非严格确定性 default 复跑本身也出现了训练指标差异。候选/默认的 50 轮自对弈指标不逐位相同，不能把这里的约 1% 观察值解释为逐牌局质量等价证明。7800 条记录的指标均为有限值。

### 相同输入的独立 GAE 计时

固定同一份真实 minibatch（64 条轨迹，Q shape=[64, 177, 35, 3]）及其 Q 值、概率、奖励、动作和步数。每种模式 3 组，每组热身 20 次、测量 100 次，同样交替顺序。两者输出逐位一致，随机状态不变。

GAE 单次 wall 时间中位数：eager 6.1159 ms，Graph 1.4289 ms，降低 76.64%，加速 4.28 倍。该计时直接证明固定输入 GAE 调度的收益；不能将单独调用的节省直接相加推算完整训练速度。完整训练仍包含自对弈、模型前后向与优化器开销及异步执行。

## 使用与结论边界

推荐使用已验证的 GAE Graph 入口，两网络继续 default。收益较小，但满足现有固定输入的数值门槛。正式训练入口可改为：

```bash
python -m experiments.vrpo_repro.train_with_gae_graph <原有 VRPO 训练参数>
```

这段命令说明入口替换，尖括号应替换为实际参数；不是已启动新的训练。仓库中已保存实现和入口。LocalServer 的可运行副本位于本次实验 reusable 目录，使用本实验 source 快照；正式服务器仓库保持干净。服务器上调用该副本需指定 Python 路径：

```bash
PYTHONPATH=/root/doudizhu/logs/vrpo-stable-speed-20261010-1826/reusable:/root/doudizhu/logs/vrpo-stable-speed-20261010-1826/source \
OMP_NUM_THREADS=6 /root/doudizhu/.venv/bin/python -m experiments.vrpo_repro.train_with_gae_graph <原有 VRPO 训练参数>
```

适用范围是本次 RTX 5070 单 GPU、FP32/TF32、同步 VRPO、静态 batch/shape。CPU、AMP、DDP、多线程并发及长期胜率尚未验证。固定输入逐位一致有助于保留本轮训练计算，但不等于已经证明长训练实战质量相同。

训练/测速单元正常结束，补充诊断也已结束，GPU 无计算进程，回到 18 MiB、0% 利用率。未启动额外长训练或申请 Colab 资源。

## 可核验证据

- [固定参数与原门槛](../logs/vrpo-stable-speed-2026-10-11/protocol.json)
- [吞吐和数值汇总](../logs/vrpo-stable-speed-2026-10-11/summary.json)
- [优化器/归一化/PPO 原始诊断](../logs/vrpo-stable-speed-2026-10-11/diagnosis-groups.json)
- [候选逐步检查](../logs/vrpo-stable-speed-2026-10-11/results/candidate/summary.json)
- [原始张量复算和更新状态检查](../logs/vrpo-stable-speed-2026-10-11/results/raw-numeric-audit.json)
- [GAE 相同输入计时](../logs/vrpo-stable-speed-2026-10-11/results/supplemental/gae-microbenchmark.json)
- [自对弈指标有限性与复跑差异](../logs/vrpo-stable-speed-2026-10-11/metric-audit.json)
- [后台状态](../logs/vrpo-stable-speed-2026-10-11/results/job-status.json)
- [显存记录](../logs/vrpo-stable-speed-2026-10-11/results/gpu.csv)
- [环境](../logs/vrpo-stable-speed-2026-10-11/results/environment.json)
- [文件哈希](../logs/vrpo-stable-speed-2026-10-11/results/artifact-manifest.json)
- [取回与资源记录](../logs/vrpo-stable-speed-2026-10-11/receipt.json)
- [可复用实现](../experiments/vrpo_repro/gae_cuda_graph.py)、[训练入口](../experiments/vrpo_repro/train_with_gae_graph.py)、[入口与 GPU 检查](../experiments/vrpo_repro/test_gae_cuda_graph.py)

结果包及 155 个文件哈希、GAE 计时的 3 个补充文件和复用入口的 5 个文件哈希均已核验。结果包 SHA256：1a372edd52b0eed76dc1ede375e557ab1a6cfaedf412e114362dac2bb6e99123。
