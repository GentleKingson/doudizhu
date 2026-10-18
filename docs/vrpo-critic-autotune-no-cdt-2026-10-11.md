# VRPO：仅 CentralCritic 使用 autotune 的复测

仅 CentralCritic 使用 `max-autotune-no-cudagraphs`、关闭 `coordinate_descent_tuning`，策略网络保持 `default` 后，仍未通过原有数值门槛。32 次实际优化 loss 和 critic 梯度全部通过，但策略梯度、部分输出和裁剪判断超限。按“通过后再测速”的条件，三组吞吐与显存对照没有启动。

## 实验条件

2026-10-11 02:17–02:19（香港时间），LocalServer RTX 5070，Python 3.12.15、PyTorch 2.14.1+cu130、CUDA 13.0。完整斗地主、v1.3 模型、FP32/TF32、AMP 关闭、rollout=256、collect=128、2 个 minibatch、policy/critic 各 4 epochs、Replay scale=16（CUDA）、Muon+AdamW、双方 base_lr=0.0004、seed=0、6 个 CPU 线程，Inductor、dynamic=False。其余模型、训练参数、冻结轨迹、初始权重和阈值与前两轮一致。

实际编译调用已记录并断言：策略网络 `mode="default"`，没有传入 autotune options；critic 展开 `max-autotune-no-cudagraphs` 预设后传入 `options={"max_autotune": true, "coordinate_descent_tuning": false}`，CUDA Graph 关闭。PyTorch 不允许同时传 mode 和 options，因此采用展开预设的方式。正式训练源码和服务器仓库没有修改。

数值检查使用严格确定性上下文、TF32 开启、CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0。两轮冻结真实轨迹更新，各执行 16 次策略更新、16 次 critic 更新，并在每轮后检查共同局面。Replay 和 minibatch 抽样序列一致。默认独立复跑的输出、loss、裁剪前梯度、更新后权重、优化器和最终 RNG 逐位一致。本轮默认基线与上一轮 37 个对应文件也逐位一致。

首次后台启动因解压多一层目录，在源码检查阶段停止，尚未进行 GPU 测试。修正解压路径后执行完整实验；初次日志保存在 startup-path-error.log，没有改数值门槛。

## 数值结果

5 项脚本测试通过。候选无 NaN/Inf，浮点参数、梯度和优化器状态为 FP32。34 个检查点中 33 个至少一个门槛未通过。

| 检查 | 两网络 autotune、CDT 开启 | 两网络 autotune、CDT 关闭 | 本轮仅 critic autotune | 原固定门槛 / 本轮结论 |
|---|---:|---:|---:|---|
| 策略梯度最大相对 L2 | 2.902569% | 1.158972% | 1.158988% | ≤0.1%，未通过 |
| 策略梯度最低余弦 | 0.999595871 | 0.999933243 | 0.999933240 | ≥0.99999，未通过 |
| critic 梯度最大相对 L2 | 0.257984% | 0.094743% | 0.091189% | ≤0.1%，通过 |
| critic 梯度最低余弦 | 0.999997165 | 0.999999611 | 0.999999639 | ≥0.99999，通过 |
| 动作概率最大绝对差 | 0.0060898066 | 0.0018337369 | 0.0018274188 | 每元素 atol=1e-5、rtol=1e-4，未通过 |
| Q 值最大绝对差 | 0.00017940998 | 0.00011926889 | 0.00011891127 | 同上，未通过 |
| 裁剪判断最大不一致比例 | 0.688600% | 0.127518% | 0.127518% | ≤0.1%，未通过 |
| 实际优化 loss | 部分更新未通过 | 32 次全部通过 | 32 次全部通过 | abs(delta) ≤ 1e-5 + 1e-4*abs(reference) |

策略实际优化 loss 最大绝对差为 5.00865e-6，critic 为 1.74642e-5，均在各自参考值对应的容差内。原脚本的 losses 门槛还检查日志统计字段，唯一超限字段为 clip_frac；它不是反向传播的目标。即使排除这个统计，策略梯度、输出和裁剪判断仍未通过。

第一步策略动作概率和 log-ratio 完全相同，critic Q 值已有最大 8.00937e-7 的差异，优势估计最大差异 7.15256e-7。第一步策略梯度相对 L2 仅 2.30695e-6；随后多次真实更新后，第 16 步策略梯度差异达到上述最大值。第二轮后 3845 个共同有效局面有 4 个裁剪判断不同（0.104031%），双方裁剪总数仍为 2068。

本轮只改变 critic 的编译配置，仍出现与上一轮近似的策略梯度偏差。这支持“critic 数值变化经优势估计与后续更新传播到策略”的解释。策略使用 default 并不保证训练后的策略权重相同。实验尚未定位具体内核，也未验证 Muon、归一化或裁剪各自放大的贡献，不能据此断言长期胜率下降或编译器存在 bug。

当前建议正式训练两网络继续使用 default。这次调整没有满足用户要求的数值等价条件，也没有取得可报告的提速证据。

## 性能门禁与资源

预定三组顺序为 default-1、autotune-1、autotune-2、default-2、default-3、autotune-3，每次 20 轮热身、30 轮测量，profiler 关闭，测量区间编译计数必须稳定。门槛未通过，六次均未启动。gpu.csv 仅记录本次数值诊断过程，不能用于稳态吞吐或训练显存对照。

systemd 单元 doudizhu-vrpo-critic-autotune-no-cdt-20261010-1818 正常结束，GPU 无计算进程，回到 18 MiB、0% 利用率；服务器正式仓库干净。没有启动长训练或申请 Colab 资源。

结论范围为一个种子、两轮冻结轨迹的数值对照。未测试稳态训练速度、三组显存或长期实战质量；数值超限不等于实战必然变差。

## 证据

- [协议与固定参数](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/protocol.json)
- [数值汇总](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/summary.json)
- [三次编译方案对比](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/comparison.json)
- [候选逐步检查](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/results/autotune/summary.json)
- [默认独立复跑](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/results/repeat/summary.json)
- [原始张量独立复算](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/results/raw-numeric-audit.json)
- [实际编译选项](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/compile_mode.py)
- [后台状态](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/results/job-status.json)
- [显存记录](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/results/gpu.csv)
- [环境](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/results/environment.json)
- [文件哈希](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/results/artifact-manifest.json)
- [取回与资源释放记录](../logs/vrpo-critic-autotune-no-cdt-2026-10-11/receipt.json)

原始 FP32 输出、GAE、loss、裁剪前梯度和共同局面保存在 results/default 和 results/autotune。压缩包及 99 个文件哈希已核验；独立 CPU 复算覆盖 34 个输出/loss 检查和 32 对梯度。结果包 SHA256：a38bd314e077243fce2ccdc9435eda3d6a2806935c0ba873f611adcfbaa0741c。
