# RTX 5070：VRPO + CentralCritic 编译模式数值对照

`max-autotune-no-cudagraphs` 未通过运行前固定的数值门槛。因此后台流程在数值阶段结束，三组吞吐/显存短跑均未启动；本次不能给出提速比例或稳态显存变化。建议暂时保留 `default`。

## 配置与执行

2026-10-11 01:58–02:00（香港时间），LocalServer，RTX 5070 12 GB，Python 3.12.15、PyTorch 2.14.1+cu130、CUDA 13.0、驱动 595.91.07。仅在独立实验目录切换 Agent 和 CentralCritic 的 `torch.compile` 模式，未改正式训练代码或公共配置。

两边都使用完整斗地主、v1.3 模型、FP32 参数与梯度、TF32 开启、AMP 关闭、rollout=256、collect=128、2 个 minibatch、策略/critic 各 4 epochs、Replay scale=16、Muon+AdamW、6 个 CPU 线程、seed=0、Inductor、`dynamic=False`。数值测试共执行两轮，每边策略和 critic 各真实更新 16 次；保持原来的 500 轮稳定学习率/裁剪/正则调度。

复用已保存的真实 FP32 轨迹，双方起始权重、空优化器状态、CPU/CUDA RNG、minibatch 排列及 Replay 抽样一致。测试采用严格确定性算法和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，同时保留 FP32/TF32。第二轮继续使用冻结轨迹，以隔离编译差异的传播；不代表常规自博弈的数据分布。

先运行默认模式，再独立复跑默认模式，最后运行候选模式。每次检查 32 次更新前的输出、GAE 目标、loss 和裁剪前梯度，并检查两次更新后共同局面的输出与裁剪判断。默认复跑还逐位比较更新后模型、优化器状态和最终 RNG。

## 数值结果

5 项脚本检查通过。默认模式复跑的输出、loss、裁剪前梯度、更新后模型/优化器状态、随机状态和抽样索引逐位一致。候选模式无 NaN/Inf，浮点参数、优化器状态和梯度均为 FP32；但 34 个检查点中有 33 个至少一个数值门槛未通过。

| 检查 | 运行前门槛 | 候选最坏结果 |
|---|---|---:|
| 输出、GAE、概率比 | 每元素 `atol=1e-5, rtol=1e-4` | 未通过；合法动作概率最大绝对差 0.00608981，Q 最大绝对差 0.00017941 |
| loss 及其日志字段 | `abs(delta) <= 1e-5 + 1e-4*abs(reference)` | 未通过；实际优化目标最大绝对差 0.0000383332 |
| 策略梯度相对 L2 | ≤ 0.001（0.1%） | 0.0290257（2.90257%） |
| 策略梯度余弦 | ≥ 0.99999 | 0.999595871 |
| critic 梯度相对 L2 | ≤ 0.001（0.1%） | 0.00257984（0.257984%） |
| critic 梯度余弦 | ≥ 0.99999 | 0.999997165，通过 |
| PPO 裁剪判断不一致比例 | ≤ 0.001（0.1%） | 更新 minibatch 最大 0.006886（0.6886%） |

实际优化目标的最大差出现在策略第 11 次更新：默认 0.0159620084、候选 0.0159236751。当次允许误差为 0.0000115962，实际误差为 0.0000383332。这里不把 clip fraction 等日志统计的差异混称为实际反向传播 loss 的差异。

最初差异很小：第一次策略梯度相对 L2 为 0.00000589855，余弦 0.999999999983，实际总 loss 相同，初始检查点通过。第 7 次策略更新已出现 1/3908 个裁剪判断不同，梯度相对 L2 升到 0.00220012。第 16 次策略更新达到上述最坏梯度误差。

固定的 3845 个有效共同局面，第一轮后有 5 个裁剪判断不同（0.1300%），第二轮后有 18 个不同（0.4681%）；两边都有裁剪、未裁剪和靠近 0.02 边界的样本。第二轮共同局面 log-ratio 最大绝对差 0.028381。原始梯度张量的全部 32 对比较已用独立 CPU 脚本复算，形状、dtype、有限性及误差汇总均吻合。

这些结果支持“更新后的训练路径出现可测分叉”，不证明候选模式损失了长期训练质量，也不证明编译器有错误。浮点运算顺序会改变舍入结果，这是 PyTorch 文档明确说明的限制；微小差异经过更新并影响 PPO 裁剪分支是本次数据支持的候选解释，尚未通过拆分内核/优化器实验定位具体原因。[PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)

## 编译与资源

本机解析出的候选选项为 `max_autotune=True`、`coordinate_descent_tuning=True`，默认模式选项为空。这个模式不启用 CUDA Graph；模式选项含义见 [torch.compile 官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)。

实际硬件为 48 SM。安装的 Inductor `is_big_gpu` 对 CUDA 使用 68 SM 门槛，GPU 模板路径受此判断限制；两种模式都记录了 `Not enough SMs to use max_autotune_gemm mode`。这只说明该 GEMM 模板路径受到限制，不能推出所有 autotune 都未运行，也不能代替吞吐实测。

数值进程总耗时分别为默认 34.82 秒、默认复跑 15.89 秒、候选 68.30 秒。这些时间包含编译、张量保存和 CPU 比较，缓存状态也不同，**不作为吞吐或纯编译时间**。显存诊断保留在 `gpu.csv`，同样不作为原计划的稳态显存对照。

原计划的三组交错短跑为：default-1、autotune-1、autotune-2、default-2、default-3、autotune-3；每次 20 轮热身、30 轮测量并要求测量期间编译计数稳定。数值门槛未通过后，这六个运行全部跳过，门槛没有放宽。

systemd 单元已正常结束，GPU 无计算进程，回到 18 MiB、0% 利用率。正式仓库仍干净。没有启动新的长训练。

## 证据与复现

- [实验协议与固定门槛](../logs/vrpo-autotune-2026-10-11/protocol.json)
- [汇总](../logs/vrpo-autotune-2026-10-11/summary.json)、[默认复跑](../logs/vrpo-autotune-2026-10-11/results/repeat/summary.json)、[候选逐步结果](../logs/vrpo-autotune-2026-10-11/results/autotune/summary.json)
- [数值脚本](../logs/vrpo-autotune-2026-10-11/numeric.py)、[编译模式切换](../logs/vrpo-autotune-2026-10-11/compile_mode.py)、[门控流程](../logs/vrpo-autotune-2026-10-11/run.py)
- [后台状态](../logs/vrpo-autotune-2026-10-11/results/job-status.json)、[显存日志](../logs/vrpo-autotune-2026-10-11/results/gpu.csv)、[环境](../logs/vrpo-autotune-2026-10-11/results/environment.json)、[硬件与 Inductor 门槛](../logs/vrpo-autotune-2026-10-11/results/hardware-guard.json)
- [原始梯度独立复算](../logs/vrpo-autotune-2026-10-11/results/raw-gradient-audit.json)、[文件哈希](../logs/vrpo-autotune-2026-10-11/results/artifact-manifest.json)、[取回与资源释放记录](../logs/vrpo-autotune-2026-10-11/receipt.json)

`results/default` 和 `results/autotune` 保存每次裁剪前的 FP32 梯度、合法动作输出、GAE、loss 和 probe 原始 `.pt` 文件。源快照包 SHA256：`57773322a74f1bde071550866f1b59627367d040abd77da2ccd7f079e492fb29`；冻结轨迹 SHA256：`172ed4eda0d6f4eff0680dac1bc78174e0afaf4e85a0b7335c42dd25faa3ea27`；结果压缩包 SHA256：`8b88d9ef71f1b9fe4c610f11416407b68f84b0eea9bcc2a1b884f24a946ac095`。压缩包及 95 个文件哈希已核验。

目前结论只覆盖一个种子、两轮冻结数据更新和当前 RTX 5070/PyTorch 环境；未测三组吞吐、稳态显存、长期训练或实战胜率。下一步若继续定位，可以在同样的数值门槛下单独隔离 `coordinate_descent_tuning` 的影响，不能把当前候选直接切入正式训练。
