"""Publish only independently audited measurements into the existing notebook."""
import json
from pathlib import Path
import re

repo = Path(__file__).resolve().parents[2]
root = repo/'logs/colab-a100-paper-2026-10-11'
audit = json.loads((root/'audit.json').read_text())
summary = json.loads((root/'summary.json').read_text())
receipt = json.loads((root/'receipt.json').read_text())
results = json.loads((root/'results.json').read_text())
assert audit['passed'] and receipt['released']
best = json.loads((root/'best-parameters.json').read_text())
runs = [r for r in results if r['label'] in best['trials']]
assert len({r['torch_version'] for r in runs}) == 1
torch_version = runs[0]['torch_version']
ram = min(r['host_min_available_mib'] for r in runs)/1024
intro = f'''## A100 High RAM 实测结果（论文参数约束，2026-10-11）

选定CPU线程 **{best['config']['threads']}**。完整replay双种子复测的合并吞吐为 **{best['trajectories_per_second']:.3f}轨迹/秒**、{best['env_steps_per_second']:.1f}环境步/秒、{best['game_steps_per_second']:.1f}实际决策步/秒。

全程NVML显存峰值 **{best['gpu_peak_mib']/1024:.3f} GiB**，保守显存余量 **{best['headroom_mib']/1024:.3f} GiB**；最低可用主机内存 **{ram:.3f} GiB**。硬件为A100-SXM4-80GB，PyTorch{torch_version}。

3组线程初筛（16热身+8计量）；选定配置以seed0/1各运行64热身+16计量，确保replay已填满131072条并经过覆盖写入。所有合格区间无新增编译，训练指标有限。计量不含编译、保存和评测，不能代表端到端含Drive保存的速度。

线程间小差异不证明统计显著优势；这是有限候选内选择，不是全局最优或牌力证明。G4论文版175.659轨迹/秒使用同一训练配方；硬件和采样路径差异仍需结合环境步吞吐解释。

实测训练未保存checkpoint。A100已关闭。GPU作业共{summary['total_seconds']/60:.2f}分钟；账户余额读数差{receipt['balance_difference']:.2f}单位，不是精确逐作业账单。

'''
rows = ['| 阶段 | CPU线程 | 轨迹/秒 | 显存峰值GiB | 保守余量GiB |', '|---|---:|---:|---:|---:|']
for r in results:
    assert r['eligible']
    rows.append(f"| {r['label']} | {r['config']['threads']} | {r['trajectories_per_second']:.3f} | {r['gpu_peak_mib']/1024:.3f} | {r['headroom_mib']/1024:.3f} |")
text = intro+'\n'.join(rows)
p = repo/'notebooks/colab_a100_highram_vrpo_centralcritic.ipynb'
n = json.loads(p.read_text())
n['cells'][2]['source'] = text
s = ''.join(n['cells'][3]['source'])
n['cells'][3]['source'] = re.sub(r'^BEST = .*$', 'BEST = '+repr(best['config']), s, flags=re.M)
for cell in n['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
p.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
p = repo/'docs/colab-a100-paper-2026-10-11.md'
s = p.read_text().replace('状态：实测进行中。旧版吞吐数字不适用于本版。', '状态：已完成；实测结果已审计，A100已释放。旧版吞吐数字不适用于本版。')
s += '\n'+text+'\n\n交付：[notebook](../notebooks/colab_a100_highram_vrpo_centralcritic.ipynb)。原始记录：[results.json](../logs/colab-a100-paper-2026-10-11/results.json)、[audit.json](../logs/colab-a100-paper-2026-10-11/audit.json)、[释放回执](../logs/colab-a100-paper-2026-10-11/receipt.json)。\n'
p.write_text(s)
print(text)
