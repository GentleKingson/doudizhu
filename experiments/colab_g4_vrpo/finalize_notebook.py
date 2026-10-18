"""Insert audited measurements into the portable Colab notebook and report."""
import json
from pathlib import Path
import re

repo = Path(__file__).resolve().parents[2]
root = repo/'logs/colab-g4-vrpo-2026-10-11'
audit = json.loads((root/'audit.json').read_text())
assert audit['passed']
best = json.loads((root/'best-parameters.json').read_text())
summary = json.loads((root/'summary.json').read_text())
results = json.loads((root/'results.json').read_text())
hardware = json.loads((root/'hardware.json').read_text())
receipt = json.loads((root/'receipt.json').read_text())
config = best['config']
rate = best['trajectories_per_second']
peak = best['gpu_peak_mib']/1024
headroom = best['headroom_mib']/1024
intro = f'''## G4 实测结果（2026-10-11）

最优已复测配置：**rollout={config['rollout']}，collect={config['collect']}，mini_batches={config['mini_batches']}，CPU threads={config['threads']}，replay_buffer_scale=16**。

双种子合并 **{rate:.3f} 条已交付训练轨迹/秒**、**{best['env_steps_per_second']:.1f} 环境步/秒**、**{best['game_steps_per_second']:.1f} 实际决策步/秒**。NVML显存采样峰值 **{peak:.3f} GiB**；按 max(NVML峰值, PyTorch reserved峰值+512MiB) 计算的最低显存余量 **{headroom:.3f} GiB**，满足至少15%总显存或3GiB的门槛。

硬件：{hardware['gpu']}，{hardware['affinity']} CPU线程。Python3.13.15，PyTorch2.11.0+cu130，CUDA13.0。训练代码固定原仓库提交；Colab软件变化可能影响结果。

共{len([r for r in results if r['label'].startswith('screen')])}组初筛和{len([r for r in results if r['label'].startswith('verify')])}次独立复测，GPU作业耗时{summary['total_seconds']/60:.2f}分钟（含安装和编译）。每组16轮热身填满replay；初筛8轮计量，复测16轮计量，以8轮为块。测量期间新增编译数为0，训练指标完整且有限。初筛只选候选，推荐吞吐用独立复测的总轨迹/总时间计算。

本次调优未保存训练checkpoint；{receipt['session']}实例已停止，释放后活动分配为0、使用速率0.00/小时。账户余额读数差{receipt['balance_difference']:.2f}单位，非逐作业精确账单。

这是有限搜索内的已复测最优，不证明全局最优、长期收敛或棋力提升。批次/minibatches/replay容量会影响学习行为。短时显存余量不能保证所有长期状态都不OOM。性能排除热身、保存和评测；Drive保存会暂停训练。
'''
rows=['| 阶段 | R/C/M/线程 | 轨迹/秒 | NVML峰值GiB | 保守余量GiB |','|---|---|---:|---:|---:|']
for r in results:
    c=r['config']; label='/'.join(str(c[k]) for k in ('rollout','collect','mini_batches','threads'))
    if r['eligible']:
        rows.append(f"| {r['label']} | {label} | {r['trajectories_per_second']:.3f} | {r['gpu_peak_mib']/1024:.3f} | {r['headroom_mib']/1024:.3f} |")
    else:
        rows.append(f"| {r['label']} | {label} | 不合格/失败 | {r['gpu_peak_mib']/1024:.3f} | — |")
table='\n'.join(rows)+'\n\nR=采样并行数，C=每轮轨迹数，M=minibatches；线程数均记录实际生效值。'
verified=['| 双种子复测配置 R/C/M/线程 | 合并轨迹/秒 | seed0 / seed1 轨迹/秒 |','|---|---:|---|']
for r in audit['verified']:
    label='/'.join(str(r['config'][k]) for k in ('rollout','collect','mini_batches','threads'))
    verified.append(f"| {label} | {r['trajectories_per_second']:.3f} | {r['seed_rates']['0']:.3f} / {r['seed_rates']['1']:.3f} |")
comparison='\n'.join(verified)+'\n\n小于短测波动的差距不代表统计显著优势。\n\n'+table
p=repo/'notebooks/colab_g4_vrpo_centralcritic.ipynb'
n=json.loads(p.read_text()); n['cells'][1]['source']=intro;n['cells'][2]['source']=comparison
s=''.join(n['cells'][3]['source']);s=re.sub(r'^BEST = .*$', 'BEST = '+repr(config),s,flags=re.M)
s=re.sub(r'^# threads .*$', '# threads 是实际使用的CPU线程数；设置1时原仓库会扩展为所有可用CPU。',s,flags=re.M)
n['cells'][3]['source']=s
for c in n['cells']:
    if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
p.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
doc=repo/'docs/colab-g4-vrpo-2026-10-11.md'
s=doc.read_text().replace('本次测试进行中，最终结果以归档的 measurement.json、summary.json 和释放回执为准。',f'已完成。最优已复测吞吐 **{rate:.3f}轨迹/秒**，显存峰值 **{peak:.3f}GiB**，保守余量 **{headroom:.3f}GiB**。G4已释放。')
s+='\n'+intro+'\n'+comparison+'\n\n## 交付与校验\n\n- [可运行notebook](../notebooks/colab_g4_vrpo_centralcritic.ipynb)\n- [最佳参数](../logs/colab-g4-vrpo-2026-10-11/best-parameters.json)\n- [原始结果](../logs/colab-g4-vrpo-2026-10-11/results.json)\n- [独立复算](../logs/colab-g4-vrpo-2026-10-11/audit.json)\n- [源代码哈希](../logs/colab-g4-vrpo-2026-10-11/source-manifest.json)\n- [释放回执](../logs/colab-g4-vrpo-2026-10-11/receipt.json)\n\nNotebook通过nbformat格式检查、各代码单元语法检查和7项离线单测（含文件保留、失败保护、定时结束、续训格式及释放顺序）。未替用户进行Drive交互授权和实际checkpoint读写，未执行完整Notebook端到端长跑或棋力评估。训练测量本身已在G4执行。\n'
doc.write_text(s)
print(p)
