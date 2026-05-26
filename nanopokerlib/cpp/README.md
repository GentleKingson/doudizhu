# nanopokerlib C++ design notes

This folder contains the C++ implementations of the `nanopokerlib` game modules exposed via
`pybindings.cpp` (pybind11).

The most important conventions for environment / training integration are:

## Header decomposition pattern

When a game/core header grows large, keep the canonical include path stable and split internals into
subfolder fragments:

- Keep `*_core.hpp`, `*_endpoint.hpp`, and `core/types.hpp` as facade entrypoints.
- Move implementation chunks to sibling folders (for example `games/<game>/core/*.hpp`,
  `games/<game>/endpoint/*_methods_*.hpp`, `ext/<module>/core/*.hpp`, `core/types/*.hpp`).
- Include fragments from the facade in order; avoid changing public symbol names or include paths.

This keeps runtime API stability while reducing per-file size and review surface.

## Header / binding layout choice

`nanopokerlib/cpp` intentionally uses an "Option 2" structure:

- Keep declaration + implementation together per component header, with implementation behind
  `#ifdef NANOPOKERLIB_IMPLEMENTATION`.
- Keep Python bindings in separate `*_bindings.hpp` companions and aggregate registrations in
  `pybindings.cpp`.

Rationale:

- Most game/runtime code is heavily template/config driven, so header-local implementation keeps
  instantiation behavior explicit and avoids brittle explicit-instantiation plumbing.
- Runtime/game headers stay usable for C++-only consumers without pulling `pybind11` into the core
  include graph.
- Pybind registration logic evolves at a different pace than game semantics; separate binding files
  keep reviews and ownership boundaries clear.

Why we do **not** fold pybind helpers into every core/type header ("Option 3"):

- It couples runtime headers to Python/ABI concerns.
- It increases compile fan-out by dragging `pybind11` includes through core paths.
- It raises duplicate-registration / ODR risk when macros are enabled across many translation units.

When to consider `include/` + `src/` split ("Option 1"):

- only if we intentionally productize a standalone compiled C++ library boundary
  (stable ABI, explicit export surface, dedicated non-template implementation units).

## Header-scope guardrails

- Keep each fragment namespace/include independent (no textual include-scope dependency).
- Do not place `#include` directives inside namespace/class blocks.
- Validate with:
  - `python scripts/check_cpp_header_scope_includes.py`

## Chance-player ids

`core/types/constants.hpp` defines two negative player ids for chance steps:

- `PLAYER_CHANCE_ENVIRONMENT = -2` (`CHANCE_ENVIRONMENT` alias)
  - Meaning: **environment-controlled** chance steps (EnvBatch handles the randomness internally).
  - Expectation: `available_actions` should contain a **single dummy action** with empty tokens:
    `ActionProto{ {}, weight = 1.0, is_commit = false }`.
  - Typical use: **private** deals / reveals that Marvel training does not need to assign value to.

- `PLAYER_CHANCE_TRAINER = -1` (`CHANCE_TRAINER` alias)
  - Meaning: **trainer-controlled** chance steps (the training routine samples an outcome).
  - Expectation: `available_actions` enumerates the chance outcomes (possibly via multi-step
    selection) and `weights` encode the **probability mass function** over those outcomes.
  - Typical use: **public** chance nodes where the critic/value function should cover the chance
    timestep(s).

Training-side note (Marvel): the sampler treats `player_id < 0` as a chance step and samples from
`available_actions.weights` rather than the agent policy. VRPO excludes hidden/environment prefix
rows from critic-loss masking, but keeps `PLAYER_CHANCE_TRAINER` steps.
In `marvel/training/common.py`, `compute_qvalue_loss(...)` applies this as
`train_mask = history_mask & (player_ids >= -1)`.

## Multi-step actions (`is_commit`)

Some games expose **token-level** multi-step actions. In those cases the environment:

- emits `available_actions.is_commit = false` for prefix-extension steps,
- keeps `player` the same until a committed action is chosen,
- updates internal prefix state (often `action_prefix_`), and
- only advances the game when `is_commit = true`.

This lets sequence models act one token at a time while still tracking “real” committed moves.

Examples:

- `hunl200` (NLHE): raising is encoded as a base-4 token sequence; intermediate raise-prefix tokens
  are non-commit until the final `r0..r3` token completes the amount (see
  `games/holdem/holdem_core.hpp`).
- `doudizhu`: complex move construction uses semantic prefixes like `333+`, `333?`, `333..` (see
  `games/doudizhu/README.md` and `games/doudizhu/doudizhu_endpoint.hpp`).

## Multi-step chance selection

Chance can also be multi-step when enumerating the full joint outcome would exceed practical action
limits.

Examples:

- Hold'em public board cards use `PLAYER_CHANCE_TRAINER` and select **rank then suit** so each step
  stays small (≤ 13 ranks, ≤ 4 suits) while representing a uniform draw from the remaining deck.
- Liars Dice `_full` variants use `PLAYER_CHANCE_TRAINER` and enumerate the full joint roll outcome
  per die index, while the non-`_full` variants keep rolling as `PLAYER_CHANCE_ENVIRONMENT` with a
  single dummy action (see `games/liarsdice/liarsdice_struct.hpp` and `games/liarsdice/liarsdice_core.hpp`).

## Episode length (`MAX_GAME_HORIZON`)

Each game exposes a `MAX_GAME_HORIZON` constant used by training code to preallocate rollout
buffers and KV caches.

In this codebase, the relevant notion of “episode length” is:

- the number of `EnvBatch.step(...)` calls between terminal frames (`StepResult.is_terminal == 1`),
- counting **all** environment steps, including:
  - chance steps (`player < 0`),
  - non-commit prefix steps for multi-step actions (`available_actions.is_commit == false`), and
  - normal committed player actions (`available_actions.is_commit == true`).

Per-game tight bounds and derivations live in the corresponding game README (for example,
`games/holdem/README.md` for `hunl200`).

## Related per-game docs

- `games/holdem/README.md`: dealing timeline, token vocabulary, and state-string endpoint.
- `games/doudizhu/README.md`: semantic infoset strings and pending-action prefixes.
- `games/liarsdice/README.md`: Liars Dice infoset strings.
