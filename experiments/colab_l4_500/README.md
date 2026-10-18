# L4 VRPO 500 campaign

This directory implements the supplied plan. The files in `plan/` are immutable
proposal inputs, not runnable training configurations or evidence of completion.
`training_ready` remains false until the recorded pre-training gates pass.

The audited source is `ba7a7eb52112a17d1fb73f79c2cebb9555cb5e0a`; the specified
upstream reference is `acdcf199cf422e2b683cb22142b8e7cb33b14460`. Implementation
started from the actual clean checkout `8a70b38317b98eb3e1eba2c1ac5ea807034b217a`.
Relative to the audited source, that checkout adds experiments and IPPO profiling,
reuses environment response getters, adds optional sampler generators (unused by
VRPO training), and canonicalizes native committed history strings. The VRPO loss
and update recipe are unchanged. The source archive and native binary fingerprint
in every generation identify the actual executed implementation.

## State and publication

Full-state mode is opt-in and restricted to single-process eager VRPO with CPU
replay. Campaign workers explicitly use OMP/MKL/intra-op threads=2 and inter-op=1.
It preserves current/EMA actor, critic, both optimizers and actual step
counts, EMA normalization state, RNG, replay ring, partial trajectories, commit
leftovers, parsed observations/outcomes, module modes and logical counts. A raw
little-endian int64 action journal reconstructs the native environment and RNG
from its original seed, including automatic resets. This training path uses the
global Python/NumPy/Torch generators; it creates no independent generators.

Replay shards own at most approximately 16 MiB of tensor storage (one trajectory
can exceed that bound for other layouts). Saving and loading never clone or
materialize a second entire replay. `CAPTURED.json` means local integrity only.
`COMPLETE.json` is emitted only after an independent Drive connector/API download
passes archive and per-file SHA256 checks. A mount-cache readback is insufficient.
Generation directories, sources, lineage and journal prefixes are immutable;
previous generations and all files outside this campaign are retained.

Workers pause at every staged generation until the coordinator verifies its
independent download and publishes COMPLETE. This wait counts in actual runtime
cost. Capture is triggered at 250/500, 900 elapsed seconds at the next completed
iteration boundary, the final continuation target, and the first full real
replay (S iterations). The first real full replay is compared shard by shard
against the paused live buffer. Repeated-trajectory capacity fixtures are only
I/O/memory tests, never training, compression or strength evidence.

Every snapshot includes the executed source, exact native extension, full
configuration, dependency inventory and existing run metadata. Final attempt
metrics/resource records are copied separately and require coordinator readback
before the job ledger can be marked complete. Integrity alone does not promote a
snapshot to FULL_STATE_VERIFIED; restoration-equivalence evidence is separate.

## Entrypoints

Run from the repository root with its locked environment:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=17 WANDB_DISABLED=true \
python -m experiments.colab_l4_500.campaign prepare \
  --candidate A0 --seed 17 --root /content/l4-work
```

`smoke` performs three genuine iterations in a separate attempt. `train` requires
`--ready <readiness.json>` and `--assignment <immutable-attempt.json>`; it checks
all evidence hashes, the execution SHA, actual runtime hostname and frozen
initial model hashes. The coordinator creates the 12 static assignments using
`assignments.py`; the only worker lock is local to its VM. No Drive/SQLite/CSV
atomic task-claim assumption is made.

Continuation uses the entire verified parent config, then applies only permitted
operational changes. In a fresh runtime, after independent download/unpacking:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=29 WANDB_DISABLED=true \
python -m experiments.colab_l4_500.campaign resume \
  --candidate A0 --seed 29 --root /content/l4-continuation \
  --ready /content/drive/MyDrive/doudizhu/l4-vrpo-500-20261018-v1/readiness.json \
  --assignment /content/new-runtime-resume-assignment.json \
  --resume /content/verified-generation --target 501
```

A real 500→501 result is required before reporting handoff complete. After that,
`--target 40000` is an absolute total target, retaining B/R/M/S, EMA, schedules,
model, optimizer, precision, seeds and environment. Changing scientific fields is
rejected. The notebook in `notebooks/colab_l4_vrpo_500_campaign.ipynb` consumes the
frozen source bundle, native runtime and environment lock from Drive.

## Evaluation

`deals.py` freezes disjoint monitor/development/primary-holdout/secondary-holdout
sets (1k/5k/10k/5k), protocol 4, seeds 100017/200029/300043/400059. Training seeds
remain 17/29/43. `evaluate.py` verifies model/opponent/deal identities, retains
per-deal IDs, both roles and raw scores, and checks exactly 2N actual games.
The existing `ci` fields remain labeled 1×SEM. `paired.py` separately computes
10,000 whole-deal bootstrap replicates with analysis seed 20261018; it averages
three trained seeds equally within each deal before resampling. Its 95% interval
is conditional on those trained models, not a training-seed population interval.

`selection.py` writes an exclusive development-choice lock before holdout; it
cannot replace an existing choice. A3 requires its 64k bridge and actual-CU
comparison for an unqualified efficiency choice. Apparent differences at the
opponent win-rate floor still require an EARLY_INCONCLUSIVE assessment rather
than a forced ranking. The evaluator refuses other candidates once a holdout
choice is locked. Missing evidence is NOT_PROVEN/INCONCLUSIVE.

## Colab CLI workaround

The installed Colab CLI 0.7.4 automation path did not reuse its cached kernel and
could fail with `Connection was lost` / a WebSocket `NoneType.sock` error.
`colab_mount.py`, run with the installed CLI's Python, attaches automation to the
cached kernel and waits for readiness; it leaves the installed package unchanged.
The native Google authorization still belongs to the user. Mount identity is
checked against folder ID `14wacgMnsJwnldEKydfQ5TFCUb5EOGTEO` using a campaign
nonce uploaded through the connected Drive account.

The user authorized only the existing CU balance. The coordinator records account
balance/rate snapshots and reserves units for preservation/handoff. Creating
`<local-work-root>/STOP_AFTER_BOUNDARY` requests a complete-iteration snapshot and
clean pause; it does not truncate an optimizer update. A paused run is not a
valid 500-iteration result. Hard quota loss still has no guaranteed save callback.
