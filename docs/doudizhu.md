# Dou Dizhu Notes

Last verified: 2026-05-26

## Evaluation Surface

The isolated evaluator entrypoint is:

```bash
python -m evaluations.run doudizhu <agent> [<agent> ...] --batch-size <N>
```

The run planner accepts these agent forms:

- `perfectdou`
- `douzero`
- `fpdou`
- `marvel=<policy_checkpoint>[?action_temp=...]`
- `marvel@yomi=<policy_checkpoint>@<belief_checkpoint>[?routine=...&search_budget=...]`

Checkpoint locators may be either local paths or `wandb:<entity>/<project>/<run_id>[:<iteration>]`.
Checkpoint resolution is intentionally centralized through `CheckpointStore.resolve_checkpoint_file_path(...)`.

## Search Routines

`routine=local-subgame` is the default search path for MARVEL+Yomi Dou Dizhu agents.
It samples Yomi belief particles, builds a local sampled subgame, evaluates leaves with the qvalue `CentralCritic` loaded from the policy checkpoint, and solves with a policy-prior-anchored PCFR response.

`routine=update-equiv` keeps the older rollout-backed update-equivalent path.
It estimates committed-action Q values from determinized rollouts, then applies the closed-form policy update:

```text
pi_new(a) proportional to pi_old(a) * exp(search_beta * q(a))
```

For `local-subgame`, omitted `search_floor` is capped by the requested budget:

```text
default_local_search_floor(search_budget) = min(16, search_budget)
```

## Runtime Shape

The host owns game batching, result aggregation, event recording, and progress output.
Agents run as socket workers through `evaluations.core.socket_worker`.
The MARVEL worker binds:

- the policy checkpoint for direct action probabilities,
- the Yomi checkpoint for belief sampling in search mode,
- the policy checkpoint again as a qvalue critic for `local-subgame`.

## Training Surface

The isolated MARVEL training entrypoint is:

```bash
python -m marvel.train --algo-name vrpo --game-name doudizhu
```

Only `ippo` and `vrpo` are registered in `marvel.training`.
The default MARVEL `game_name` is `doudizhu`.

The isolated Yomi training entrypoint is:

```bash
python -m yomi.train --game-name doudizhu --policy-checkpoint-path <policy_checkpoint>
```

Yomi uses the shared MARVEL runtime for distributed setup, logging, checkpointing, and rollout sampling.
