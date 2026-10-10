# FP32 可复现性与 Critic BF16 实验回执

结论：FP32 基线可复现；仅 Critic 训练前向 BF16 数值通过，性能 **PERF_NO_GO**，保留 FP32+TF32。训练参数和数值门槛未改，实验补丁已撤回。

任务状态：**已关闭（2026-10-10，用户要求）**。`PERF_NO_GO` 为本轮最终判定，不再重复相同 BF16 实验，也不继续 P4、长期训练或棋力评估。只有用户明确要求重新开启研究时才恢复执行。

保留全部原始证据及既有哈希索引，原始判定、门槛、日志、张量、状态和脚本均不修改或删除；本地与 LocalServer 档案继续保留。

## 根因与运行控制

原 FP32 重复对照的首步前向、权重、24 次排列/Replay 索引及最终 CPU/CUDA RNG 相同。首步梯度差异仅在 state_tokens_embed、state_player_embed、action_tokens_embed 三个 Embedding 权重，最大约 1.28e-9；随后经真实更新传播，在第二轮产生 38/3845（0.9883%）个裁剪判定分歧。

当前 PyTorch 的 CUDA Embedding 反向对重复索引累加。保存的三个非确定性 Inductor 反向核使用 tl.atomic_add；严格确定性版本改走 aten.index_put_。原 token ID 配固定合成上游梯度的算子重复测试，在 eager/compiled 默认模式均不逐位一致，严格模式均一致。该测试用于定位，不用于数值门槛或边界覆盖。

检查入口使用 experiments/vrpo_repro/repro_runtime.py 的确定性上下文，覆盖编译、首次 AOT、反向及全部真实更新。Python 启动前设置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=6
```

上下文采用 torch.use_deterministic_algorithms(True, warn_only=False) 和 cuDNN 确定性设置，正常/异常出口恢复开关。TF32 仍为 high/enabled；TrainConfig.deterministic 和 deterministic_disable_tf32 字段未改，两组使用同一外部运行控制。直接调用 Trainer 的检查入口须先沿用 marvel.utils.configure_torch 初始化精度。[PyTorch 确定性 API](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html)、[可复现性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。

两个独立 FP32 进程各两轮、各 16 次 Policy + 16 次 Critic 真更新，3,884 个保存张量、155,689,374 个元素逐位一致，覆盖完整裁剪前梯度、前向样本、模型/优化器状态和 RNG/索引。两轮 log-ratio 差异及裁剪分歧均为 0。

## Critic BF16 数值与吞吐

使用原 v1.3 默认模型、Muon+AdamW、Inductor、seed=0、OMP=6、rollout=256、collect=128、mini_batches=2、Replay scale=16、Policy/Critic epochs=4/4。初始 bootstrap、Agent/Sampler、GAE/loss 保持 FP32；参数、梯度、优化器状态和 Replay 浮点存储均为 FP32，只有 Critic 训练前向采用 BF16。编译实验全生命周期的 backward_pass_autocast=off 随上下文退出恢复。

环境：RTX 5070 12227 MiB，Python 3.12.15、torch 2.14.1+cu130、CUDA 13.0、驱动 595.91.07。源基点 acdcf199cf422e2b683cb22142b8e7cb33b14460 加诊断覆盖包。

冻结批次 SHA256：172ed4eda0d6f4eff0680dac1bc78174e0afaf4e85a0b7335c42dd25faa3ea27。门槛文件原字节保留于 experiments/vrpo_repro/thresholds-critic-only.json，SHA256：00d017aa38030cffe4d058a0ca508a6650202104b84ffdff0b234ae350ac54af。

| 两轮真实更新后的指标 | 结果 | 原门槛 |
|---|---:|---:|
| abs Δlog-ratio p99 | 0.000804543 | ≤ 0.01 |
| clip 判定分歧 | 4/3845（0.1040%） | ≤ 0.5% |
| 逐 minibatch 最小 Policy 梯度余弦 | 0.999912777 | ≥ 0.99 |
| 逐 minibatch 最小 Critic 梯度余弦 | 0.999976706 | ≥ 0.99 |
| A/B 实际边界附近样本 | 107/106 | 各 ≥ 10，且均有 clipped/unclipped |
| 非有限值 | 0 | 0 |

初始 eager/compiled、FP32 重复及 BF16 A/B 数值门槛全部通过；候选完成两轮各 16+16 次真实更新。编译初始合法 Q 误差 p99 为 0.000204332、QLambda target 为 0.000309030，低于原 0.001 门槛。

数值通过后执行三对同期独立进程，顺序 A1/B1/B2/A2/A3/B3，各 8 轮热身 + 6 轮完整 profiler-off 计时。热身结束均为 39 个 unique graphs，测量期没有新增 Dynamo/AOT 编译。完成游戏计数来自实际 terminal，交付轨迹单独记录。

| 配对 | FP32 完成游戏/s | BF16 完成游戏/s | 增益 |
|---|---:|---:|---:|
| 1 | 35.6862 | 37.5169 | 5.1301% |
| 2 | 35.7034 | 37.4800 | 4.9761% |
| 3 | 35.7139 | 37.4760 | 4.9339% |

三对均正增益，中位数 **4.9761% < 10%**，未通过原性能门禁。比较两组均采用严格确定性模式，历史默认模式的速度没有进入分母。首次性能试跑遗漏 configure_torch，被精度自检拦截并独立归档，未计入配对；修复入口后从第一对重测全部六次。

## 证据、自检与边界

轻量判定、配对 CSV、冻结门槛和已撤回的候选补丁保留于 experiments/vrpo_repro。补丁基于本次诊断源码，作为原始证据归档；生产训练默认没有 BF16 开关或精度变化。

完整 236 项哈希索引、113 个张量文件、96 份裁剪前梯度、每轮状态、原始日志和复演脚本保存在忽略的 runs/profiles/vrpo-critic-bf16-repro-2026-10-10。本地与 LocalServer 原始档案均核验，服务器绝对路径为 /root/doudizhu/runs/profiles/vrpo-critic-bf16-repro-2026-10-10；该目录的 reproduce_commands.txt 保留作历史记录，任务关闭后不再执行其中的相同实验。2 GB 原始档案不纳入 Git。

```bash
WANDB_DISABLED=true python -m pytest -q \
  marvel/tests/test_profiling.py experiments/vrpo_repro/test_repro_runtime.py
```

确定性开关只在实验上下文中生效。性能门禁失败后停止 P4 trace/额外稳定性序列，未做长期训练或棋力评估；确定性模式有额外运行开销，可复现性结论限于本轮硬件、软件和控制条件。
