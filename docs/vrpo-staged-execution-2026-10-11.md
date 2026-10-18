# MARVEL VRPO 分阶段执行与首轮 GPU 验证

2026-10-11，香港时间。按批准的 LocalServer RTX 5070、累计 30 GPU 分钟硬上限完成首轮、续跑及 K1 确认。S1a、K1、E1、G1 的已声明数值合同均通过。K1 两对新独立 AB/BA 的 wall 降幅为 **3.6532%**，预注册 95% 配对区间 **[3.2343%, 4.0703%]**，达到本次指定配置下的 `PASS_ENGINEERING`；E1/G1 仍为单对粗筛 `INCONCLUSIVE_SINGLE_PAIR`。累计占用 **1614.460400 秒（26 分 54 秒）**，GPU 已释放，长期质量未验证。

首批在 11:17:52 启动、11:21:19 因验证工具报错停止，占用 207.261949 秒。修复后，用户授权使用原额度续跑，未追加预算；续跑在 11:35:44.843642 启动、11:52:20.470034 完成，占用 995.626392 秒。随后用户再次授权用余量优先确认 K1，本次增加 411.572059 秒。三个阶段均使用覆盖整个进程组的 systemd watchdog；预算按完整占用时段累计，不按 GPU 利用率折算。最终剩余 **185.539600 秒（约 3 分 6 秒）**未使用。远端主工作区保持 clean，代码仍未提交或合入。

## K1 独立确认与工程判定

本次复用已完成的正常 A/A 作为先行校准样本，把剩余预算用于 **两对新独立进程 AB、BA**，不将已用于候选选择的旧 K1 单对结果并入确认区间。20 轮预热、30 轮连续测量及原训练配置均不变；每个进程 seed=0，使用正常非确定性模式。A/A 门和工作量门在本次新数据之前冻结；旧 A/A 对制定门限可见，因此它不是新的留出检验，也不足以估计可靠噪声方差。

冻结规则为：A/A 相对 wall 差不超过 1%；配对环境步数、游戏步数相对差不超过 2%，完成局数差不超过 0.2%；轨迹长度、合法动作数量归一化直方图的总变差分别不超过 0.10、0.02。配置、正常运行设置、交付轨迹及更新次数必须一致，正式块编译图保持 39→39；配对正式块平均温度差不超过 5°C，轮询温度不超过 80°C，显存不超过 10 GiB，不允许其他 GPU 计算进程。非有限指标、任一合同或资源检查失败立即停止，不补跑。

原 A/A 的 wall 差为 0.6961%，环境步数差 −1.0244%、游戏步数差 −1.3867%，长度／动作分布总变差为 0.06797／0.00343，满足上述校准门。新 K1 结果如下，A 为基线、B 为候选：

| 独立对 / 次序 | A wall / 秒 | B wall / 秒 | wall 降幅 | 环境步数变化 | 游戏步数变化 |
|---|---:|---:|---:|---:|---:|
| 1 / AB | 56.410028 | 54.330695 | 3.6861% | −0.2357% | −0.3333% |
| 2 / BA | 56.292949 | 54.254973 | 3.6203% | −0.1577% | −0.2839% |

两对均通过工作量门：完成局数差分别为 0、−0.0520%，长度总变差为 0.06068、0.07474，动作分布总变差为 0.00536、0.00528，平均温度差为 3.64°C、0.73°C。四个计时块各交付 3840 条轨迹、Policy/Critic 各更新 240 次，全部记录的指标有限，正式块无重编译。生产源码未变，复用已通过的 K1 严格数值、51 项 CUDA 缓存合同及共享调用者兼容证据；启动前核验了 448 份源码／证据／控制文件的哈希。

统计单位为独立进程对：`d=log(wall_A/wall_B)`，固定 n=2，以 `mean(d) ± t(0.975,df=1)·s/√2` 构造区间，再转换为 `1-exp(-d)`。点估计 **3.6532% ≥ 0.5%**，95% 区间下界 **3.2343% > 0**；数值、工作量、正常性能、兼容性与资源条件均通过，因此判为 **`PASS_ENGINEERING`，限当前设备、配置和预注册口径**。两对的 Student-t 区间依赖独立且近似正态的 log 比值假设，n=2 无法检验该假设或稳定估计方差；这不是跨种子、跨设备或长期棋力保证。真实采样工作量仅在声明容差内可比，并非逐样本一致。

四个进程启动及预热耗时分别为 A1 45.968、B1 44.741、B2 44.702、A2 46.080 秒，包含本次私有缓存副本的命中状态，不能外推全冷缓存启动成本。观察器与上轮相同。84 个轮询样本观察到峰值 **7680 MiB**、最高 **69°C**。12:13:15.221829–12:20:06.793121 完成服务；计费从 launch 与 journal 中较早的启动单调时间开始，实际 **411.572059 秒**。本轮 RuntimeMaxSec=588、终止宽限 5 秒，加历史占用的最坏边界为 1795.888341 秒，未突破 1800 秒。

最终核账为 `1202.888341 + 411.572059 = 1614.460400` 秒。GPU 计算进程为空、利用率 0%，没有额外运行。归档首次遇到 tar 因在被打包目录内创建输出而报目录变化；仅将归档输出移至目录外重新打包，未重启 GPU，最终 45 份证据文件全部通过本地哈希核验。

证据：[K1 确认汇总](../logs/vrpo-k1-confirm-2026-10-11/result-summary.json)、[冻结协议](../logs/vrpo-k1-confirm-2026-10-11/protocol.json)、[判定原件](../logs/vrpo-k1-confirm-2026-10-11/results/verdict.json)、[最终预算和设备状态](../logs/vrpo-k1-confirm-2026-10-11/results/final-system-status.json)、[CPU 门禁自检](../logs/vrpo-k1-confirm-2026-10-11/cpu-self-check.txt)。协议 SHA256 为 `f517b282b887146e69eade1cdb414ce094459c6a41ef59561dd53ceba9595795`；全部四个进程原始 CSV、指标、日志及 systemd journal 已保存本地。

## 前两批结果与预算记录

按 E1 CUDA 合同→G1 合同及数值→正常 A/A→K1→E1→G1 独立测速顺序执行，未叠加候选。E1 的空 batch、batch 1/5 与原生环境输出生命周期共 **4 项 CUDA 合同通过**。G1 的 4 项测试通过，其中 1 项真实 CUDA 测试以 11 次调用覆盖最多 8 个图、第 9 个新 key 回退 eager 及旧输出不被覆盖；另外 3 项为控制流测试。G1 冻结两轮数值的 **34 项比较逐位一致**，Policy/Critic 各 16 次更新、24 次随机抽样，完整模型/优化器/RNG 一致，非有限值为 0。

正常模式关闭严格确定性运行设置，A/A 两个连续计时块分别为 **56.394 秒、56.002 秒**，第二次 wall 减少 0.696%。随后三项候选均完成独立比较，预先固定的执行次序为 K1 AB、E1 BA、G1 AB：

| 候选 | 基线 A / 秒 | 候选 B / 秒 | 观察到的 wall 降幅 | 环境步数变化 |
|---|---:|---:|---:|---:|
| K1 | 56.394662 | 54.246432 | +3.8093% | −0.6279% |
| E1 | 56.044661 | 56.104639 | −0.1070% | +0.2383% |
| G1 | 56.350728 | 55.742337 | +1.0797% | −0.7868% |

8 个性能进程均为 20 轮预热、30 轮连续计时；每个计时块交付 3840 条轨迹，Policy/Critic 各更新 240 次，编译图计数保持 **39→39**，状态均为 `MEASURED`。G1 性能进程调用 wrapper 800 次、仅 1 个图，首次捕获发生在预热期。数值阶段保留严格设置，两种模式的证据不混用。

A/A 与各 A/B 的真实环境步数、轨迹长度及合法动作分布不同，单对结果没有置信区间。正常 A/A 未预设工作量完全相同或噪声百分比停止门；本次固定计数、运行配置、编译及资源检查通过，但不能据此声称严格同工作量或把小幅 wall 差异归因为优化。**K1 值得优先复验；E1 未显示明确收益；G1 的小幅改善仍可能受工作量差异影响。** 原工程门和质量门不变。

所有性能模式使用相同观察器，保留 95,185,920 字节 GPU 副本，在计时块之后复制直方图到 CPU；上述计时和显存包括观察器开销。各模式使用独立编译缓存，历史冷热状态不同，因此启动/预热时间差不能归因为候选改动。8 个计时块峰值 allocated 约 6.649–6.659 GiB、reserved 约 7.244–7.262 GiB。约 5 秒轮询的 202 个资源样本中，观察到峰值 **7734 MiB**、最高 **70°C**；不等于硬件连续绝对峰值。

续跑成功后 transient unit 已卸载，事后 `systemctl show` 的默认零时间不能用于核账。最终时长取同一 invocation 的 journal `Started`→`Deactivated successfully` 单调时钟差：`(49466922326−48471295934)/1e6 = 995.626392` 秒；停止状态文件注明证据方法，原始卸载后查询结果另行保留。累计为 `207.261949 + 995.626392 = 1202.888341` 秒。结束时计算进程为空、利用率 0%，未再启动 GPU 任务。

证据入口：[汇总及全部测量值](../logs/vrpo-gpu-continuation-2026-10-11/result-summary.json)、[运行状态](../logs/vrpo-gpu-continuation-2026-10-11/results/job-status.json)、[最终系统状态](../logs/vrpo-gpu-continuation-2026-10-11/results/final-system-status.json)、[原始 systemd journal](../logs/vrpo-gpu-continuation-2026-10-11/results/systemd-journal.jsonl)、[续跑冻结协议](../logs/vrpo-gpu-continuation-2026-10-11/protocol.json)、[续跑源码清单](../logs/vrpo-gpu-continuation-2026-10-11/package-manifest.json)。续跑封存包的 1082 份文件中 1066 份与首批相同；16 份变化限于执行控制、测试及比较辅助工具，生产数值路径未变。38 份新增原始 tensor 文件保留于远端 `/root/doudizhu/logs/vrpo-staged-continuation-20261011-1134`，本地保存摘要、日志和哈希清单，已核验 78 份本地产物。

## 首批停止与修复记录

冻结输入 A/A、真实采样 A/A、K1、E1 各完成 34 个比较项，每个进程两轮合计 16 次 Policy、16 次 Critic 更新及 24 次 Replay/minibatch 随机抽样。全部原始比较逐位一致，梯度相对 L2 为 0，完整模型/优化器与 Python、NumPy、CPU/CUDA RNG 检查通过。真实采样每次比对 432 个事件，包括 119 次环境转换、118 次动作记录、55 次缓存重置及 8 次 Replay 采样。

CUDA 套件完成 K1 的 48 项重置组合和 3 项缓存后续输出测试，随后在 E1 的空 batch 用例报错：测试辅助函数将末维 stride 为 0 的空 Long 张量转换为 uint8 视图，触发 `RuntimeError`。此处未完成数值比较，当时结论为**验证工具失败、合同覆盖未完成**；不能据此判 E1 数值回归，也不能把首批 E1 CUDA 套件记为通过。

停止后仅修改共享测试辅助函数：保留 dtype/shape 断言，零元素张量直接返回。新增零 stride 空张量回归测试；相关 CPU 测试 **57 passed**。当时未在 GPU 上重验，随后按用户续跑授权完成上述 4 项 E1 CUDA 合同。首批封存的源码包没有被修改。首批 GPU 轮询观察到的峰值显存为 **7596 MiB**，最高温度 59°C；轮询峰值不等于连续测得的硬件绝对峰值。

证据入口：[运行状态](../logs/vrpo-gpu-firstpass-2026-10-11/results/job-status.json)、[系统停止状态](../logs/vrpo-gpu-firstpass-2026-10-11/results/final-system-status.json)、[冻结协议](../logs/vrpo-gpu-firstpass-2026-10-11/protocol.json)、[CUDA 失败日志](../logs/vrpo-gpu-firstpass-2026-10-11/results/cuda-contracts.log)。154 份原始 tensor 文件保留在远端实验目录 `/root/doudizhu/logs/vrpo-staged-firstpass-20261011-1116`；[产物清单](../logs/vrpo-gpu-firstpass-2026-10-11/results/result-artifacts.json)记录哈希。本地已保存摘要、轨迹事件、日志、协议和封存源码。停止后的测试修复单独保存在 [postflight-empty-oracle.patch](../logs/vrpo-gpu-firstpass-2026-10-11/postflight-empty-oracle.patch)。

## 审阅结论

方案的单变量顺序和算法边界可采用，实际执行需要以下校正：

1. GAE 实现和历史档案已在本地找回。155/95/96/99 项原件均存在，每组 43 份源码与协议匹配；无需重做已有微基准。历史约 1% 观察仍不满足新连续块协议下的工程采用门。
2. 原两轮数值 runner 把 collect 替换为冻结 rollout，绕过环境、Sampler 和 KV reset。它不能证明 E1/K1 路径被执行；本次增加独立真实 Sampler 的 CPU/GPU 检查补齐覆盖。
3. 本轮补齐实际 PG 分支/梯度系数、Python/NumPy RNG、逐调用输入身份与零/极小梯度合同。服务器 PyTorch 2.14 的 clip 边界导数与本地 2.13 不同，观察器已改为记录生产 max/clamp 的实际 autograd 系数，并经服务器 CPU 预检。斗地主有效玩家只可能是 `-2/0/1/2`，真实有效 `-1` 记 N/A；共享 loss 的 `-1/-2` 语义保留合成合同测试。
4. padding 数据来自首批较短轨迹，数值 harness 两轮重复同一批。该 fixture 的共同尾部约 48.04%，不能外推稳态收益。Critic 每轮 8 次完全相同且同顺序的整批前向重复数为 0；不进入 P1/C1。
5. GAE cache 原本无界。本次以最多 8 个图、超限新 key 回退 eager 补齐控制流边界；CUDA 输出生命周期、图上限/回退及当前配置显存已验证，未外推全部形状与长期运行。

计划的 wall 降幅区间 `[1-exp(-L), 1-exp(-U)]` 顺序正确。BF16 的旧 10% 门槛、三个 autotune 失败和稀疏等待 STOP 均保留，未重开研究。

## 基线与候选合同

B0 是 `6f3070686f4e62cb128cc28fcc8886fd2d0edac2`。本次共同源码基线 B 选择 **执行前 L0 工作区**，其中 13 份已有改动/未跟踪文件已保存原文、diff 和 SHA256；GAE wrapper 在独立 K1/E1 比较中保持关闭。既有 IPPO 等无关文件按原哈希验证未动。

后续 A/B 的两侧都应用相同 S1 计时补丁，再分别应用单个候选。原逐轮同步 CSV 不能作为新分母，必须重测 A。四份独立补丁都已在重建的 B0+L0 上单独检查可应用，首次封存时组合文件与工作区逐字节一致。GPU 部署包另记录全部 1082 份文件的哈希和独立候选差异；停止后的测试修复为单独增量。详细来源见 [baseline_manifest](../logs/vrpo-staged-execution-2026-10-11/baseline_manifest.json)、[candidate_manifest](../logs/vrpo-staged-execution-2026-10-11/candidate_manifest.json) 和 [GPU 部署清单](../logs/vrpo-gpu-firstpass-2026-10-11/package-manifest.json)。

| 工作包 | 本次改动/验证 | 状态 |
|---|---|---|
| S0 | 保存原工作区；恢复原件、源码映射和 STOP 清单 | 本地核对完成 |
| S1a | 冻结输入与真实 Sampler 严格 A/A，PG/RNG/输入身份/完整状态 | `PASS_NUMERIC`，限已声明两轮合同 |
| S1b | 新增 `--benchmark --benchmark-contiguous`；仅块首尾同步，输出真实计数、启动/预热 wall、显存和编译变化 | CPU 合同及 8 个 CUDA 性能进程检查通过 |
| K1 | 两轮真实采样逐位通过；51 项 CUDA 缓存合同通过；两对新 AB/BA 通过预注册性能及工作量门 | `PASS_ENGINEERING`，限当前配置及两对统计口径 |
| E1 | 同次转换 rewards/observations/available_actions 从 2/2/3 次降至 1/1/1；两轮真实采样逐位通过 | `PASS_NUMERIC`；4 项专属 CUDA 合同通过；性能 `INCONCLUSIVE_SINGLE_PAIR` |
| G1 | 恢复原实现；8 图封顶，新 key 超限回退原 eager；两轮冻结数值逐位通过 | `PASS_NUMERIC`；4 项测试通过（含 1 项 CUDA）；性能 `INCONCLUSIVE_SINGLE_PAIR` |
| P0/C0 | 还原原索引，区分 fixture 与稳态；不重排 minibatch | 生产准入 `EVIDENCE_MISSING` |
| A1/P1/C1 | 尚未满足各自的完整数值/数据入口 | 未实现 |
| J1/Q1/R1 | K1 已满足本次单项工程门；组合、质量与发布未执行 | 未进入 |

S1 文件合同扩到 `marvel/config.py`、`marvel/train.py` 以暴露默认关闭的新计时开关，另改诊断器、其测试和既有文档；未改训练循环、epochs、损失、精度、Replay/RNG 调度或优化器。K1/E1/G1 仅改各自原文件与测试。它们是未提交的工作区候选，尚未决定正式采用。

数值门沿用原阈值：输出/loss `atol=1e-5, rtol=1e-4`，梯度相对 L2 ≤0.001、余弦 ≥0.99999、clip 分歧 ≤0.001；本轮候选另要求原始数据与完整状态逐位相同。每项单对 A/B 粗筛已完成，该阶段无置信区间、不能宣称工程通过；随后 K1 的新独立确认单独判定如上。原 wall 工程门与质量门没有放宽。

## 验证与证据

完整命令和输出保存于 [final-cpu-tests.txt](../logs/vrpo-staged-execution-2026-10-11/final-cpu-tests.txt) 及各候选记录。本机 Python 3.11.15、PyTorch 2.13.0；历史 CUDA 档案使用另一套版本，CPU 结果不跨版本外推。为运行真实环境测试，使用既有源码和缓存 pybind11 headers，在临时目录由 Apple clang 离线编译了两个斗地主 translation unit，未改 C++ 或项目依赖声明。缺少的现有项目依赖 tqdm/tyro 仅安装到临时测试目录。

```sh
WANDB_DISABLED=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/private/tmp/vrpo-staged-native/deps \
python3 \
/private/tmp/vrpo-staged-native/run-cpu-tests.py -q \
  marvel/tests/test_env_batch.py marvel/tests/test_cache_reset.py \
  marvel/tests/test_profiling.py experiments/vrpo_repro/test_repro_runtime.py \
  experiments/vrpo_repro/test_gae_cuda_graph.py
```

初始候选合并结果：**78 passed、3 skipped，8.42 秒**。新增执行工具后的本地预检为 **99 passed、58 skipped**；PG 边界观察器修正后的服务器 CPU 预检 **22 passed**。计时测试验证模型/优化器/RNG/计数不变，模拟 CUDA API 核对首尾 fence、peak reset、CSV 和异常合同。首批 GPU 停止后的空张量比较器修复另通过 **57 项 CPU 测试**。续跑预算控制及相关工具的本地检查 **44 passed**，服务器 CPU 预检 **60 passed**；E1 CUDA 4 项、G1 测试 4 项随后通过。不同阶段测试数量不相加作为独立覆盖数。

- [历史证据与 P0/C0/G1 审阅](../logs/vrpo-staged-execution-2026-10-11/evidence-review.md)
- [S1a 真实覆盖与最小补缺](../logs/vrpo-staged-execution-2026-10-11/s1a-review.md)
- [K1 验证记录](../logs/vrpo-staged-execution-2026-10-11/k1-result.md)、[E1 验证记录](../logs/vrpo-staged-execution-2026-10-11/e1-result.md)、[G1 验证记录](../logs/vrpo-staged-execution-2026-10-11/g1-result.md)
- [数值/性能/资源合同](../logs/vrpo-staged-execution-2026-10-11/protocol.json)

## 下一阶段与回退

本次授权的 K1 确认已完成。建议将 K1 单独进入最小工程变更审阅，保留接口、回退和本次证据；当前判定不等于自动合入或默认部署。E1/G1 继续保持待定，不叠加。若需要比当前两对条件区间更强的统计结论，应另定更多配对、当期 A/A 和跨种子确认；本次不会继续消耗余量或追加预算。已通过的数值证据仅在生产源码、环境和输入哈希不变时复用。

独立补丁位于结果目录：`s1-contiguous.patch`、`k1-reset-with-tests.patch`、`e1-getter.patch`、`g1-bounded-graph.patch`。确认文件仍匹配 candidate manifest 后，可先 `git apply --reverse --check <该补丁>` 再逆向应用；只撤回本次工作，不用 `git reset --hard` 覆盖原用户改动。GAE 本次补丁回退到原 L0，默认入口继续使用 eager。原档案和新的证据均保留。

未执行候选叠加、完整保存/评测集成或长期多种子训练，未 commit、push、开 PR 或合入；长期棋力与 time-to-target 均为 `NOT_VERIFIED`。本次完成后未使用剩余额度追加运行。
