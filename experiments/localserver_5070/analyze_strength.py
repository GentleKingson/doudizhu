"""Audit paired game outcomes and summarize the completed run's training metrics."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'logs/localserver-5070-eval-2026-10-12'


def wilson(wins, n):
    z = 1.959963984540054
    p = wins / n
    scale = 1 + z*z/n
    center = (p + z*z/(2*n)) / scale
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / scale
    return [max(0., center-half), min(1., center+half)]


def mean_ci(values):
    mean = statistics.mean(values)
    half = 1.959963984540054 * statistics.stdev(values) / math.sqrt(len(values))
    return {'mean': mean, 'ci95_normal': [mean-half, mean+half], 'n_paired_deals': len(values)}


def matchup(path):
    official = json.loads((path/'summary.json').read_text())['matchups']
    assert len(official) == 1 and official[0]['ok']
    roles = defaultdict(dict)
    for line in (path/'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['kind'] != 'episode':
            continue
        row = event['body']
        assert 'ckpt_576.pt' in row['right_agent']
        role = 'peasants' if row['left_view_role'] == 'as_landlord' else 'landlord'
        index = int(row['episode_id'])
        assert index not in roles[role], 'Duplicate episode'
        win = 1-int(row['landlord_win']) if role == 'peasants' else int(row['landlord_win'])
        score = -float(row['landlord_score']) if role == 'peasants' else float(row['landlord_score'])
        assert win in (0, 1) and math.isfinite(score)
        roles[role][index] = (win, score)
    assert set(roles) == {'landlord', 'peasants'}
    assert set(roles['landlord']) == set(roles['peasants']) == set(range(1,1025))
    result = {}
    for role, games in roles.items():
        wins = sum(w for w, _ in games.values())
        result[role] = {'games': len(games), 'wins': wins, 'win_rate': wins/len(games),
                        'win_rate_ci95_wilson': wilson(wins, len(games))}
    ids = sorted(roles['landlord'])
    for column, name in [(0, 'role_average_win_rate'), (1, 'role_average_landlord_scale_gain')]:
        # Pair the same initial deal across seats; do not treat both passes as independent.
        result[name] = mean_ci([(roles['landlord'][i][column]+roles['peasants'][i][column])/2 for i in ids])
    assert abs(result['role_average_win_rate']['mean'] + official[0]['left_view_metrics']['role_avg_win_rate']['mean'] - 1) < 1e-12
    assert abs(result['role_average_landlord_scale_gain']['mean'] + official[0]['left_view_metrics']['role_avg_avg_gain']['mean']) < 1e-12
    return result


def main():
    assert abs(wilson(50, 100)[0] - (1-wilson(50, 100)[1])) < 1e-12
    assert abs(wilson(0, 100)[0]) < 1e-12 and abs(wilson(100, 100)[1]-1) < 1e-12
    assert mean_ci([.5, .5])['ci95_normal'] == [.5, .5]
    metrics = defaultdict(dict)
    with (ROOT/'logs/localserver-5070-2026-10-12/train/metrics/metrics.csv').open() as f:
        for row in csv.DictReader(f):
            step, value = int(row['step']), float(row['value'])
            assert math.isfinite(value) and step not in metrics[row['key']]
            metrics[row['key']][step] = value
    windows = [(1,50), (101,150), (251,300), (401,450), (477,526), (527,576)]
    summary = {}
    for key, rows in metrics.items():
        assert set(rows) == set(range(1,577))
        summary[key] = {'windows': {f'{a}-{b}': statistics.mean(rows[i] for i in range(a,b+1)) for a,b in windows},
                        'max_step': max(rows, key=rows.get), 'max': max(rows.values()), 'final': rows[576]}
    assert all(metrics['buffer/replay_buffer_size'][i] == 131072 for i in range(64,577))
    payload = {'perspective': 'ckpt_576 EMA agent_state_dict, greedy action_temp=0, no search',
               'confidence': '95% Wilson per role; paired-deal normal interval for role average; conditional on this checkpoint and opponent',
               'training': summary,
               'matchups': {p.name: matchup(p) for p in sorted(OUT.iterdir()) if p.is_dir() and (p/'summary.json').exists()}}
    (OUT/'analysis.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload['matchups'], indent=2))


if __name__ == '__main__':
    main()
