# VRPO + CentralCritic：关闭 coordinate_descent_tuning 的复测

关闭 `coordinate_descent_tuning` 后仍未通过上一轮固定的数值门槛。因此按“通过后再测速”的要求，三组吞吐和显存短跑没有启动。实际优化 loss 和 critic 梯度已经通过，剩余问题主要是策略更新后的输出与梯度，以及少量裁剪判断差异。建议暂时保留正式训练的 `default`。

## 保持不变的条件

2026-10-11 02:07–02:09（香港时间），LocalServer，RTX 5070，Python 3.12.15、PyTorch 2.14.1+cu130、CUDA 13.0。模型与训练源码、冻结输入、初始权重、随机序列和门槛均与[上一轮实验](vrpo-autotune-2026-10-11.md)一致。

完整斗地主、v1.3、FP32 参数/梯度/优化器状态、TF32 开启、AMP 关闭、rollout=256、collect=128、2 个 minibatch、policy/critic 各 4 epochs、Replay scale=16、Muon+AdamW、seed=0、6 个 CPU 线程、Inductor、`dynamic=False`。没有改正式训练代码、模型结构、SM 门槛或优化器。

候选预设仍为 `max-autotune-no-cudagraphs`。PyTorch 不接受同时传 `mode` 和 `options`，因此实验展开预设后只覆盖 `coordinate_descent_tuning=False`；两个模型实际收到的 options 都是 `{"max_autotune": true, "coordinate_descent_tuning": false}`。CUDA Graph 默认关闭且脚本进行了断言。默认对照仍使用 `mode="default"`。

数值检查沿用严格确定性运行时、FP32/TF32、`CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `PYTHONHASHSEED=0`。两轮冻结真实轨迹更新，每边策略、critic 各真实执行 16 次优化器更新；每次检查裁剪前梯度、合法动作输出、GAE 和 loss，并在每轮后检查共同局面。Replay 抽样与 minibatch 排列一致。

## 检查结果

5 项脚本检查通过。默认模式独立复跑逐位一致；本轮默认基线与上一轮的 34 个检查文件、两次更新后模型/优化器状态及最终随机状态共 37 个文件也逐位一致。候选没有 NaN/Inf，所有浮点参数、梯度和优化器状态为 FP32。

| 检查 | coordinate descent 开启（上轮） | 关闭（本轮） | 固定门槛 / 本轮结论 |
|---|---:|---:|---|
| 策略梯度最大相对 L2 差异 | 2.90257% | 1.15897% | ≤0.1%，未通过 |
| 策略梯度最低余弦 | 0.999595871 | 0.999933243 | ≥0.99999，未通过 |
| critic 梯度最大相对 L2 差异 | 0.257984% | 0.0947428% | ≤0.1%，通过 |
| critic 梯度最低余弦 | 0.999997165 | 0.999999611 | ≥0.99999，通过 |
| 合法动作概率最大绝对差 | 0.00608981 | 0.00183374 | 每元素 `atol=1e-5, rtol=1e-4`，未通过 |
| Q 值最大绝对差 | 0.000179410 | 0.000119269 | 同上，未通过 |
| 实际优化 loss | 部分更新未通过 | 32 次更新全部通过 | `abs(delta) <= 1e-5 + 1e-4*abs(reference)` |
| 裁剪判断最大不一致比例 | 0.688600% | 0.127518% | ≤0.1%，未通过 |

本轮策略实际优化 loss 最大绝对差为 `4.86523e-6`，出现在第 11 次更新；critic 实际优化 loss 最大绝对差为 `1.77622e-5`，出现在第 14 次更新。后一项参考值为 0.552512169，允许误差为 `6.52512e-5`，所以仍通过。此处“实际优化 loss”分别指 PG 加 KL 正则、Q-lambda loss。

原始脚本的 `losses` 门槛也检查日志统计字段，其中唯一未通过的字段为 `clip_frac`；它不是实际反向传播目标。不能将这些日志统计失败描述为“实际优化 loss 失败”。即使忽略它，策略梯度和输出门槛仍未通过。

第一次更新前合法动作概率、log-ratio 完全相同，策略梯度相对 L2 仅 `2.30695e-6`。最大策略梯度差出现在第 16 次真实更新。固定的 3845 个共同有效局面，第一轮后裁剪判断完全一致，第二轮后有 4 个不同（0.104031%）；虽然双方裁剪总数都是 2068，发生裁剪的样本集合已经不同。第二轮共同局面 log-ratio 最大绝对差为 0.00513411。

本轮 34 个检查点中 33 个至少一个门槛未通过。独立 CPU 脚本已重新核验全部 34 个输出/loss 检查、32 对梯度的 shape、dtype、有限性、余弦、相对 L2，以及全部门槛判定。

关闭开关后，本次冻结数据中的多项误差减小，但关闭该选项不足以取得所要求的数值等价。当前实验没有定位剩余差异对应的具体内核，也没有证明长期训练质量或实战胜率变差；只有一个种子、两轮冻结轨迹和一次候选数值运行。

## 后续流程与资源

计划与上一轮保持一致：default-1、autotune-1、autotune-2、default-2、default-3、autotune-3；每次 20 轮热身、30 轮测量，测量期间编译计数必须稳定。因为数值门槛未通过，六次运行全部跳过，门槛没有放宽。

`gpu.csv` 记录的是本次数值诊断的显存占用，不能代替稳态训练显存对照。本次没有新的吞吐数据或显存提升结论。

systemd 单元 `doudizhu-vrpo-autotune-no-cdt-20261010-1807` 正常结束，GPU 无计算进程，回到 18 MiB、0% 利用率，正式仓库仍干净。没有启动新的长训练。

## 证据

- [协议、参数与固定门槛](../logs/vrpo-autotune-no-cdt-2026-10-11/protocol.json)
- [汇总](../logs/vrpo-autotune-no-cdt-2026-10-11/summary.json)、[与上轮比较](../logs/vrpo-autotune-no-cdt-2026-10-11/comparison.json)
- [候选全部检查](../logs/vrpo-autotune-no-cdt-2026-10-11/results/autotune/summary.json)、[默认独立复跑](../logs/vrpo-autotune-no-cdt-2026-10-11/results/repeat/summary.json)
- [原始数据独立审计](../logs/vrpo-autotune-no-cdt-2026-10-11/results/raw-numeric-audit.json)、[实际编译选项脚本](../logs/vrpo-autotune-no-cdt-2026-10-11/compile_mode.py)、[数值检查脚本](../logs/vrpo-autotune-no-cdt-2026-10-11/numeric.py)
- [后台状态](../logs/vrpo-autotune-no-cdt-2026-10-11/results/job-status.json)、[显存日志](../logs/vrpo-autotune-no-cdt-2026-10-11/results/gpu.csv)、[环境](../logs/vrpo-autotune-no-cdt-2026-10-11/results/environment.json)
- [文件哈希](../logs/vrpo-autotune-no-cdt-2026-10-11/results/artifact-manifest.json)、[资源释放与取回记录](../logs/vrpo-autotune-no-cdt-2026-10-11/receipt.json)

`results/default` 与 `results/autotune` 保留各步原始 FP32 输出、GAE、loss、裁剪前梯度和共同局面 `.pt` 文件。结果包 SHA256 为 `202a4438c368f68bba0ce69192aae88258868ddc99e30258056bd71da6302e83`，压缩包及 96 个文件哈希均已核验；训练源码、冻结输入和原始脚本哈希也已核验。
