# K1 最小工程变更与合入审阅

2026-10-11。结论：**可合入所审阅的 K1 独立变更，当前仅整理和审阅，未提交、暂存、合入或发布。** 本次 GPU 使用量为 0。

## 改动说明与范围

终局重置原先对整个 K/V 张量执行广播 masked-fill；K1 先取需要清空的玩家行，再沿 batch 维原位清零。生产修改仅 `marvel/model/cache.py::KVCache.reset` 的三行替换。张量布局仍为 `[layers, batch×players, heads, time, dim]`，接口、长度/位置元数据处理、mask 转换和错误处理不变。

独立交付仅四个文件：

- `marvel/model/cache.py`：三行生产替换。
- `marvel/tests/test_cache_reset.py`：保留已有的 41 项 K1 CPU 合同测试，包括空张量比较辅助函数修复。
- 本审阅记录。
- [机器可读证据](evidence/vrpo-k1-2026-10-11.json)：配置、逐对测量、CPU 结果、文件与原始证据哈希。

不包含 S1 计时实现、E1 getter、G1 图缓存、训练配置或依赖变化。此次没有写入任何已有工作区文件，也没有把整个工作区差异打成提交。审阅期间保护快照检测到三份 Colab 文件被并行工作更新；它们未被本次操作写入或回退，具体路径记录在交付清单中。

## 审阅结果

未发现需要在合入前修复的问题。检查以当前斗地主训练/搜索的固定缓存布局、稀疏到全量终局重置为范围。

1. **清零维度和别名正确。** 构造器将玩家 batch 放在 dim=1；`index_fill_` 同时覆盖所有层、head、时间和通道。每层 `key_cache/value_cache` 继续引用原存储，未选中行逐位不变；不会把含 NaN 的选中行变成乘零后的 NaN。
2. **输入与边界行为保持。** device/dtype 转换、末维 squeeze、batch 检查及空 mask 提前返回均保留。非连续 mask、bool/int/float、NaN mask、重复重置、异常维度的行为与冻结旧实现对照；长度、位置和 step mask 不引入新语义。
3. **调用者无需改动。** Marvel 在 cache rebuild 中清空全部玩家行，在终局路径用 `mask.repeat_interleave(P)` 清空对应环境的全部玩家。Kagami bootstrap 和 Yomi collector 复用这些真实方法；prefill、rebuild、reset→append、cache-only 和满窗输出通过对照。
4. **成本和范围清楚。** 新增 `nonzero` 索引分配及 CUDA 同步可能性，不宣称零同步，也不从局部操作量推断速度。已有真实训练 wall 对照包含整条调用路径。没有新持久缓存、配置开关、导出、依赖或 checkpoint 格式变化。

## 本次 CPU 回归与隔离验证

从 HEAD `6f3070686f4e62cb128cc28fcc8886fd2d0edac2` 在临时目录提取源码，仅覆盖 K1 生产文件及其测试；没有复制当前工作区的 S1/E1/G1 或其他未提交修改。确认实际导入来自该临时目录。

Python 3.11.15、PyTorch 2.13.0，使用 macOS CPU 构建，`torch.version.cuda is None`；显式设置 `CUDA_VISIBLE_DEVICES=`。复用此前由未变更 C++ 源码构建的临时斗地主原生扩展，未修改项目依赖。

**97 passed，0 failed，0 skipped，2.21 秒。**

| 覆盖 | 数量 |
|---|---:|
| K1 缓存语义、真实斗地主采样、Marvel/Kagami/Yomi 调用兼容 | 41 |
| Kagami 斗地主引擎 | 26 |
| Kagami 斗地主搜索 | 15 |
| Kagami local-subgame 搜索 | 9 |
| Marvel 评测接线 | 6 |

标准依赖和本地原生扩展就绪后，可从隔离源码根目录运行：

```sh
CUDA_VISIBLE_DEVICES= WANDB_DISABLED=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
python -m pytest -q -ra marvel/tests/test_cache_reset.py kagami/tests \
  evaluations/doudizhu/tests/unit/test_marvel_agent_wiring.py
```

本机依赖加载及实际执行入口另存于 `logs/vrpo-k1-engineering-2026-10-11/run_cpu.py`；日志和 JUnit XML 同目录保留。

## 保留的性能证据

GPU 数值/性能证据来自此前获批运行，本次没有重跑。K1 的两轮真实 Sampler 数值比较共 34 项逐位一致，完整模型/优化器/RNG 一致；51 项 K1 CUDA 缓存合同通过。所审阅生产文件和测试的 SHA256 与此前封存 K1 源码一致。

| 新确认配对 | 基线 wall / 秒 | K1 wall / 秒 | wall 降幅 |
|---|---:|---:|---:|
| AB | 56.410028 | 54.330695 | 3.6861% |
| BA | 56.292949 | 54.254973 | 3.6203% |

按冻结 n=2、df=1 的 paired log-wall Student-t 口径，合并降幅 **3.6532%**，95% 区间 **[3.2343%, 4.0703%]**，满足点估计至少 0.5%、区间下界大于 0 的工程门。四个正式块均为 30 轮、3840 条轨迹、Policy/Critic 各 240 次更新，编译图 39→39；工作量及资源门通过。

GPU 对照基线为冻结 L0 加双方共有的计时/观察器，A/B 生产差异只有 K1；本次 CPU 验证基线则是 HEAD 加 K1。不能把它写成此次在纯 HEAD+K1 上重新测得 GPU 性能。两对样本不能检验区间的分布假设，正常 A/A 是前轮校准样本；收益范围限 RTX 5070 与已声明配置，不外推全冷缓存启动、跨设备或长期棋力。

原始证据保留于 `logs/vrpo-k1-confirm-2026-10-11/` 和 `logs/vrpo-gpu-firstpass-2026-10-11/`。本次审阅包另收录 K1 数值摘要、原 CUDA 合同日志、确认原始归档及 CPU 记录；日志中的 E1 测试辅助函数历史失败不是 K1 失败，也未纳入 K1 补丁。

## 补丁与回退

`logs/vrpo-k1-engineering-2026-10-11/k1-minimal.patch` 是相对上述 HEAD 的四文件独立补丁；当前工作区已经包含 K1，**不要再次直接应用**。补丁已经在独立目录验证正向应用、结果文件哈希及完整逆向恢复。

需要撤销 K1 运行时行为时，使用同目录的 `k1-runtime.patch`，先检查再逆向应用：

```sh
git apply --reverse --check logs/vrpo-k1-engineering-2026-10-11/k1-runtime.patch
git apply --reverse logs/vrpo-k1-engineering-2026-10-11/k1-runtime.patch
```

该回退只恢复 reset 的三行旧实现，保留测试、审阅和历史证据；已在临时目录演练。若检查失败或目标文件后来发生其他改动，应先审查冲突，不能强制覆盖；不使用 `git reset --hard` 或整文件 checkout 撤销用户改动。原 checkpoint 和训练数据无需迁移。

**合入结论：可合入这个独立 K1 补丁；没有阻塞项。** 不将当前整个脏工作区视为可合入对象。未检查新的 GPU 运行、完整长训棋力或其他设备，未提交或合入。
