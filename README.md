# Dou Dizhu

This repository is a one-commit isolation of the Dou Dizhu training, evaluation, and search stack from `nanogamekits`.

It keeps the pieces needed to train Dou Dizhu MARVEL/Yomi checkpoints and run Dou Dizhu matchups with Kagami search-backed MARVEL agents:

- `evaluations/doudizhu`: scenario planner, host loop, socket agents, runtime protocol, summaries, and tests.
- `evaluations/core`: shared queue, controller, worker, resource, progress, and W&B orchestration used by the evaluator.
- `kagami/games/doudizhu`, `kagami/core`, and `kagami/search`: the Dou Dizhu inference engine plus `update-equiv` and `local-subgame` search routines.
- `marvel`: Dou Dizhu policy training and inference support, with only `ippo` and `vrpo` registered as training algorithms.
- `yomi`: Dou Dizhu belief-model training and inference support.
- `nanopokerlib`: the local C++/pybind runtime needed for Dou Dizhu state parsing, reconstruction, and environment stepping.
- `scripts/setup-perfectdou.sh`, `scripts/install-perfectdou.sh`, and `scripts/install-fpdou.sh`: optional external baseline bootstraps.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel cmake ninja
python -m pip install -e .[dev]
python -m pip install -v --no-build-isolation -e nanopokerlib
```

Optional W&B support:

```bash
python -m pip install -e .[wandb]
```

Optional PerfectDou baseline:

```bash
bash scripts/setup-perfectdou.sh --name perfectdou37
```

## Train

MARVEL defaults to `game_name=doudizhu` in this artifact. Only `ippo` and `vrpo` are registered:

```bash
python -m marvel.train --algo-name vrpo --game-name doudizhu
python -m marvel.train --algo-name ippo --game-name doudizhu
```

Yomi trains a Dou Dizhu belief model from a frozen MARVEL policy checkpoint:

```bash
python -m yomi.train \
  --game-name doudizhu \
  --policy-checkpoint-path /path/to/policy.pt
```

Checkpoint locators may be local paths or `wandb:<entity>/<project>/<run_id>[:<iteration>]`; all checkpoint resolution flows through `CheckpointStore.resolve_checkpoint_file_path(...)`.

## Run

Direct evaluator entrypoint:

```bash
python -m evaluations.run doudizhu \
  perfectdou \
  marvel=/path/to/policy.pt?action_temp=0 \
  --num-games 1024 \
  --batch-size 128 \
  --workers-per-gpu 2 \
  --eval-data external/PerfectDou/eval_data_10k.pkl \
  --logdir runs/doudizhu
```

Search-backed MARVEL agent specs use `marvel@yomi=<policy>@<belief>`:

```bash
python -m evaluations.run doudizhu \
  perfectdou \
  'marvel@yomi=/path/to/policy.pt@/path/to/belief.pt?routine=local-subgame&search_budget=512&search_beta=1' \
  --num-games 1024 \
  --batch-size 128
```

Supported search routines:

- `routine=local-subgame`: critic-backed local subgame PCFR with a policy prior.
- `routine=update-equiv`: rollout-backed update-equivalent search.

## Validation

Useful smoke checks after setup:

```bash
python -m compileall evaluations kagami marvel yomi
python -m pytest evaluations/doudizhu/tests kagami/tests -q
```

The integration tests that compare against PerfectDou require the optional `external/PerfectDou` checkout and its Python 3.7 environment.

## License

MIT. See [LICENSE](LICENSE).
