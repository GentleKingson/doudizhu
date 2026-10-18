# VRPO FP32/TF32 性能测量

同一诊断器也支持 `--algo-name ippo`，记录 `ippo.*` 阶段并按 IPPO 更新循环验证 trace。对照时统一模型、精度、rollout/collect batch、mini-batches、策略 epoch、种子及线程设置；VRPO 的独立 critic 和 replay 保留为算法差异。

RTX 5070 上的三组同 batch 对照和独立阶段计时见 [2026-10-11 测量与提速建议](ippo-vrpo-samebatch-2026-10-11.md)。

诊断默认关闭，训练超参数、模型、优化器和编译设置沿用现有配置。`--benchmark` 保存完整迭代的墙钟基线；`--profile` 保存一个完整迭代的 CPU/CUDA trace。两次运行应使用相同计算配置，分别保存结果。

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--benchmark` | 关闭 | 每轮开始/结束同步 CUDA，记录 wall、Sampler 实际计数与最大分配显存 |
| `--benchmark-contiguous` | 关闭 | 与 `--benchmark` 同用，预热后只在整个测量块首尾同步；不得与 `--profile` 同用 |
| `--profile` | 关闭 | 要求 CUDA 硬件以及 CPU/CUDA profiler activities；记录 trace、算子表和阶段表 |
| `--profile-trace-dir` | `<logging_dir>/profiles` | 本次测量的独立目录；已有目录不会被覆盖 |
| `--profile-warmup-iterations` | 3 | 两种测量的编译预热轮数，按实际完成的迭代计数 |

编译预热之后才启动 profiler 的 `wait=1, warmup=1, active=1, repeat=1` 周期，每个完整 training iteration 调用一次 `prof.step()`。编译预热应延长到耗时稳定、日志不再持续重编译；profiler 的 warmup 周期不能替代它。[PyTorch 编译模型 profiling 指南](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)

以下使用已在 RTX 5070 上运行的 VRPO/Muon 调优参数。仓库默认 `collect=2048, replay_scale=64` 的 replay 预分配约 59.46 GiB，12 GB GPU 会 OOM。默认训练配置仍保持原值。

```bash
OMP_NUM_THREADS=6 WANDB_DISABLED=true python -m marvel.train \
  --algo-name vrpo --game-name doudizhu --device cuda --seed 0 \
  --rollout-batch-size 256 --collect-batch-size 128 --mini-batches 2 \
  --replay-buffer-scale 16 --compile-backend inductor \
  --param-stats-interval -1 --checkpoint-interval -1 --evaluation-interval -1 \
  --total-iterations 14 --profile-warmup-iterations 8 \
  --benchmark --run-name vrpo-wall-new \
  --profile-trace-dir runs/profiles/vrpo-wall-new

OMP_NUM_THREADS=6 WANDB_DISABLED=true python -m marvel.train \
  --algo-name vrpo --game-name doudizhu --device cuda --seed 0 \
  --rollout-batch-size 256 --collect-batch-size 128 --mini-batches 2 \
  --replay-buffer-scale 16 --compile-backend inductor \
  --param-stats-interval -1 --checkpoint-interval -1 --evaluation-interval -1 \
  --total-iterations 11 --profile-warmup-iterations 8 \
  --profile --run-name vrpo-profile-new \
  --profile-trace-dir runs/profiles/vrpo-profile-new
```

没有启用 AMP、BF16/FP16 或关闭编译。此配置保持默认 v1.3 模型、Policy/Critic 各 4 个 epoch、Muon/AdamW；每轮各 8 次训练 mini-batch 更新。参数统计按该调优配置关闭；短跑评估和 checkpoint 均关闭，以避免末轮保存污染计时。OMP 等环境变量需在启动 Python 前设置。

输出目录包含 `config_snapshot.json`（配置、源码 SHA256、硬件、精度设置和环境）、`wall_baseline.csv` 或 `profile_wall.csv`、`findings.md`。Profile 模式另外保存 `trace.json`、`cpu_top.txt`、`cuda_top.txt`、`phase_summary.csv` 和 `trace_validation.json`。支持单进程；CPU benchmark 可用于逻辑 smoke，CUDA 性能结论须在 CUDA 硬件上测量。

`completed_games_delta` 来自 Sampler 的真实 terminal 计数，`game_steps_delta` 是 commit 动作步数；`collect_batch_size` 是实际交付的轨迹数。完成游戏与交付轨迹的逐轮差异来自 commit buffer 中的剩余轨迹。吞吐汇总应使用累计真实计数除以累计 wall 秒数。

阶段 CPU 时间来自 CPU annotation，CUDA kernel 时间按原始 trace 的 launch correlation 归属，包含 autograd 工作线程发射的 kernel。不能用 GPU annotation 的持续时间代替 kernel 总时间，也不能按主线程嵌套关系漏掉 backward。原生算子表保留 PyTorch 输出，其中同名 CPU/GPU 注解及算子/kernel 可能分别列出；阶段归因以 `phase_summary.csv` 为准。

CPU、CUDA 与嵌套阶段时间不能直接相加当作 wall；CUDA kernel 时间不包含 memcpy/memset。`iteration_wall_share` 留空，只有额外互斥、边界同步的粗粒度诊断才可填相应占比。Profile 的 wall 单独保存，不能替代 profiler-off 基线。Trace 缺少真实 CPU 算子、CUDA kernel、必要阶段、kernel correlation，或采集期间出现编译事件，均判为 `PROFILE_INVALID`。

`compiled_unique_graphs` 逐轮记录 Dynamo 的累计图数。比较时检查测量窗口内图数不再增加，并检查运行日志中的重编译；否则延长预热后重新测量。真实游戏完成数与交付轨迹吞吐分别报告。

异常退出会停止 profiler 并关闭采样 scope。最小检查：

```bash
WANDB_DISABLED=true python -m pytest -q marvel/tests/test_profiling.py
```

检查包括 CPU 训练结果/计数一致性、关闭时无 profiler scope/CUDA 同步、CPU-only profile 拒绝、异常清理、跨工作线程 kernel 归属，以及真实 CUDA trace 的阶段和计数。CUDA 检查需要 CUDA/CUPTI。[PyTorch Profiler API](https://docs.pytorch.org/docs/2.14/profiler.html)

## 连续块测量

在原 profiler-off 命令中加入 `--benchmark-contiguous`，并为每次运行指定新的输出目录。先完成 `profile_warmup_iterations` 个实际迭代，再从下一轮开始连续测到 `total_iterations`。测量包含正常日志、评测和保存；这些功能是否开启由原配置决定。块内不增加 CUDA fence、显存峰值 reset、逐轮计时或 CSV flush。原 `--benchmark` 的逐轮同步接口保持不变。

新模式输出单行 `wall_contiguous.csv`，包含整块 wall、每实际迭代摊销 wall、真实交付轨迹/完成游戏/环境步/commit 步计数和吞吐、块内峰值 allocated/reserved 显存及首尾编译图计数。`startup_and_warmup_ms` 从诊断器初始化前开始，到第一次正式迭代前为止，包含配置快照、运行时初始化和全部预热，不单独声称是编译耗时。块内新增编译或恢复后不足三轮测量会标 `INCONCLUSIVE`；中途异常不发布部分块。`MEASURED` 仅表示测量完成，不是数值、性能或质量通过。

新旧 CSV 的同步和 I/O 口径不同，不能互作 A/B 分母。候选与基线都需要重新用相同连续块模式采样，另行核对运行日志中的重编译、设备干扰、完整工作量和实际优化器更新次数。GPU 配对规模、预算与统计方法在开始性能实验前冻结。CPU 测试验证两算法的模型/优化器/RNG/计数保持一致，以及首尾 fence、异常、恢复与编译变化处理；不代替 CUDA 性能实验。
