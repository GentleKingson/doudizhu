# DouDiZhu (PerfectDou-compatible)
Last verified: 2026-05-01

This directory contains the C++ implementation for:

- `nanopokerlib.doudizhu`
- `nanopokerlib.doudizhu_small`
- `nanopokerlib.doudizhu_tiny`

Key files:

- Game core + dealing + step semantics: `doudizhu_core.hpp`
- Token vocabulary: `doudizhu_tokens.hpp`
- Observation channels (`NUM_CHANNELS = 288`): `doudizhu_channels.hpp`
- Semantic infoset string endpoint: `doudizhu_endpoint.hpp`

## Seats & player ids

- Player `0`: Landlord (`L`)
- Player `1`: Peasant1 (`P1`)
- Player `2`: Peasant2 (`P2`)
- Chance player id: `-2` (environment-controlled chance steps)

## Dealing timeline (tokens after `INIT`)

`reset()` returns an `INIT` frame (`obs_tokens[*] = [INIT]`) with `player = -2`.

Then the environment runs a fixed dealing script:

- The deck is shuffled once at `reset()` and then consumed in-order.
- Each private deal round deals to players in seat order `0 → 1 → 2`.
- The final `3` cards of the shuffled deck are the landlord public cards.
- During all chance steps, the only legal action is the empty action (`action.tokens = []`).

1. **Private deal rounds** (`BASE_HAND_CARDS` steps; `17` in full `doudizhu`)
   - `player = -2` on every step
   - Each step deals **one private card per player**
   - Observation tokens (per-player, private; `obs_tokens[p]` differs across players):
     - Landlord: `[PLAYER(L), RANK(x), HOLD]`
     - Peasant1: `[PLAYER(P1), RANK(x), HOLD]`
     - Peasant2: `[PLAYER(P2), RANK(x), HOLD]`

2. **Landlord public cards** (`3` steps)
   - `player = -2` on every step
   - Each step reveals **one** of the 3 landlord public cards to **all** players and adds it to the landlord hand
   - Observation tokens (broadcast; `obs_tokens[0] == obs_tokens[1] == obs_tokens[2]`):
     - `[PLAYER(L), RANK(x), HOLD]`

3. **Transition to play** (`1` step)
   - `player = 0` (landlord to act)
   - Observation tokens (broadcast): `[PLAYER(L)]`
   - `available_actions` becomes the landlord’s legal action-prefix tokens, as before

Gameplay then proceeds with the existing multi-step action-prefix semantics (`available_actions[*].is_commit`
controls when a full discard commits).

## Tokens

DouDiZhu uses a single rank-token set for both dealing and action construction.

### Rank symbols (per variant)

Card strings are written as rank symbols (suits are ignored). Each character means "one card of this rank"
and duplicates mean multiple copies. Parsing is case-insensitive and ignores whitespace.

Valid symbols depend on the game variant:

- `doudizhu`: `3456789TJQKA2BR` (`B`=black joker, `R`=red joker)
- `doudizhu_small`: `9TJQKA2BR`
- `doudizhu_tiny`: `QKA2BR`

### Token vocabulary

Token ids are exposed under `nanopokerlib.doudizhu.token.*`:

- Special: `PAD_TOKEN = 0`, `INIT_TOKEN = 1`
- `PLAYER(p)`: `p ∈ {0,1,2}`
- `ActionClass`: `{DISCARD_SOLO, DISCARD_PAIR, DISCARD_TRIO, DISCARD_QUAD, DISCARD_STRAIGHT, DISCARD_PAIR_SISTERS, DISCARD_TRIO_CHAIN, DISCARD_ROCKET}`
- `RANK(r)`: `r ∈ [0, NUM_RANKS)` (rank symbols above)
- `SeqFrom`, `SeqTo`: straight/chain endpoints
- Kicker-type/count tokens: `NoKicker`, `SoloKicker(xk)`, `PairKicker(xk)`
- (Kicker ranks are selected via extra `DISCARD_SOLO` / `DISCARD_PAIR` steps with `RANK(r)`.)
- `PASS`
- `HOLD` (observation-only marker used during dealing)

### Token id layout (`nanopokerlib.doudizhu` / full ranks)

For the full game (`NUM_RANKS = 15`, `NUM_PLAYERS = 3`) the ids/ranges are:

- `0`: `PAD`
- `1`: `INIT`
- `2..4`: `PLAYER(0..2)` (`PLAYER_OFFSET = 2`)
- `5..12`: `ActionClass` (`ACTION_CLASS_OFFSET = 5`)
- `13..27`: `RANK(0..14)` (`RANK_OFFSET = 13`, ranks `3456789TJQKA2BR`)
- `28..38`: `SeqFrom(0..10)` (`SEQ_FROM_OFFSET = 28`, ranks `3..K`)
- `39`: `SeqTo(0)` (**deprecated / unsupported**; unreachable because sequences have min length `2`)
- `40..50`: `SeqTo(1..11)` (`SEQ_TO_OFFSET = 39`, ranks `4..A`)
- `51..60`: kicker type/count (`KICKER_TYPE_OFFSET = 51`)
  - `51`: `NoKicker`
  - `52..56`: `SoloKicker(x1..x5)`
  - `57..60`: `PairKicker(x1..x4)`
- `61..75`: deprecated/reserved (legacy `SoloKicker(rank)`; unused/unreachable)
- `76..88`: deprecated/reserved (legacy `PairKicker(rank)`; unused/unreachable)
- `89`: `PASS` (`PASS_TOKEN = 89`)
- `90`: `HOLD` (`HOLD_TOKEN = 90`)
- `NUM_TOKENS = 91`

(Token ids are stable for backwards compatibility: `SeqTo(0)` and the legacy kicker-rank tokens stay reserved but never
appear in legal actions.)

(For `doudizhu_small` / `doudizhu_tiny`, the same offsets exist but the numeric ranges change with
`NUM_RANKS`; use `nanopokerlib.doudizhu_small.token.*` / `nanopokerlib.doudizhu_tiny.token.*`.)

### What tokens appear in observations?

`obs_tokens[p]` is always a **token list** (variable length). What you see depends on the step:

- `INIT` (from `reset()` and after terminal auto-reset): `player = -2`, `obs_tokens[*] = [INIT]`
- Chance private deal (first `BASE_HAND_CARDS` steps after `INIT`): `player = -2`
  - `obs_tokens[0] = [PLAYER(0), RANK(x), HOLD]`
  - `obs_tokens[1] = [PLAYER(1), RANK(x), HOLD]`
  - `obs_tokens[2] = [PLAYER(2), RANK(x), HOLD]`
- Chance public landlord reveal (next `3` steps): `player = -2`
  - `obs_tokens[*] = [PLAYER(0), RANK(x), HOLD]` (broadcast)
- Transition to play (1 step): `player = 0`
  - `obs_tokens[*] = [PLAYER(0)]` (broadcast)
- Playing phase (all later steps): `player ∈ {0,1,2}`
  - `obs_tokens[*]` is broadcast and equals the **last chosen action-prefix token list** (it always
    starts with `PLAYER(player_who_acted)`).
  - `available_actions[i].tokens` are the **next** legal action-prefix token lists and
    `available_actions[i].is_commit` tells you whether picking it completes a full discard.
  - After a commit, the returned `player` advances to the next actor, but `obs_tokens[*]` still
    contains the committed action tokens from the previous actor (so the observation is “what just
    happened”, while `player` is “who acts next”).

## Channels (NUM_CHANNELS = 288)

Per-player `obs_channels[p]` concatenates:

1. **Self remaining cards**: `54`
2. **Landlord public cards**: `54` (the 3 broadcast cards; stays set after reveal)
3. **Landlord discarded cards**: `54`
4. **Peasant1 discarded cards**: `54`
5. **Peasant2 discarded cards**: `54`
6. **Number of bombs played**: `15` zero-anchored cumulative prefix flags (`0..14`, counting bombs + rockets)
7. **Active player**: `3` one-hot (`L/P1/P2`; all zeros during chance steps)

The 54-card vectors use a fixed full-deck embedding (independent of `doudizhu_small` / `doudizhu_tiny`).

Offsets are exposed under `nanopokerlib.doudizhu.channel.*` (e.g. `SELF_OFFSET`, `DISCARD_OFFSET`,
`BOMB_COUNT_OFFSET`, `TURN_OFFSET`).

### Channel index ranges (`nanopokerlib.doudizhu`)

`obs_channels[p]` is length `288` with the following ranges:

- `0..53`: self remaining cards (`SELF_OFFSET = 0`)
- `54..107`: landlord public cards (`LANDLORD_PUBLIC_OFFSET = 54`)
- `108..161`: landlord discards (`DISCARD_OFFSET + 0 * 54`)
- `162..215`: peasant1 discards (`DISCARD_OFFSET + 1 * 54`)
- `216..269`: peasant2 discards (`DISCARD_OFFSET + 2 * 54`)
- `270..284`: bomb count cumulative prefix thresholds (`BOMB_COUNT_OFFSET = 270`, `0..14`)
- `285..287`: active player one-hot (`TURN_OFFSET = 285`, `L/P1/P2`; all zeros during chance steps)

Bomb-count channel examples:

- `0` bombs: `1000...`
- `1` bomb: `1100...`
- `2` bombs: `1110...`

Historical behavior change:

- Before `2026-03-15`, this slice was one-hot: `0 -> 1000...`, `1 -> 0100...`, `2 -> 0010...`.
- Current behavior is cumulative to expose a prefix-style monotone count signal while keeping the channel width unchanged.
- This can change inference for existing checkpoints once bombs appear because the observation contract is different even though the tensor shape is unchanged.
- To restore the historical one-hot behavior in a local revert, change `Channels::encode_bomb_count_channels(...)` in `doudizhu_channels.hpp` to set only `encode_bomb_count(count)`, and keep `decode_channel(...)` labels aligned with the one-hot semantics.

### 54-card index mapping (fixed full-deck embedding)

Every 54-slot card vector uses `FULL_RANKS = "3456789TJQKA2BR"` with:

- For normal ranks `3..2` (`13` ranks): `rank_index * 4 + copy_index`
  - `3`: `0..3`, `4`: `4..7`, ..., `2`: `48..51`
- `B`: `52`
- `R`: `53`

Counts are represented as a multiset: if you have `k` copies of a rank, the first `k` slots for that
rank are set to `1` (e.g., 3 copies of `Q` sets `Q#0,Q#1,Q#2`).

## Semantic infoset strings

`nanopokerlib.doudizhu`, `nanopokerlib.doudizhu_small`, and `nanopokerlib.doudizhu_tiny` expose:

- `parse_infoset_str(infoset_str: str, pending_action: str = "") -> History`
- `parse_infoset_strs(infoset_strs: list[str], pending_actions: list[str]) -> HistoryBatch`
- `expand_belief_prefix_str(infoset_str: str, belief_str: str = "") -> dict`
- `expand_belief_prefix_strs(infoset_strs: list[str], belief_strs: list[str] = []) -> dict`
- `reconstruct_env(history_str: str, pending_action: str = "", seed: int = 0) -> tuple[Environment, StepResult]`
- `reconstruct_envs(history_strs: list[str], pending_actions: list[str] = [], seed: int = 0) -> tuple[EnvBatch, StepBatch]`
- `determinize_history_str(infoset_str: str, belief_str: str) -> str`
- `determinize_history_strs(infoset_strs: list[str], belief_strs: list[str]) -> list[str]`
- `infoset_str_from_history_str(history_str: str) -> str` (active-player only)
- `infoset_str_from_history_strs(history_strs: list[str]) -> list[str]` (active-player only)
- `Environment.advance_history_str(history_str: str) -> str`
- `EnvBatch.advance_history_strs(history_strs: list[str]) -> list[str]`
- `Environment.belief_supervision() -> dict`
- `EnvBatch.belief_supervision() -> dict`
- `Environment.get_available_action_strs() -> list[str]`
- `EnvBatch.get_available_action_strs() -> list[list[str]]`

These helpers provide a **semantic** (rank-string) representation for reconstructing an infoset
and replaying a history without having to materialize all token-level intermediate steps.

### Belief API Notes

- `belief_supervision()` returns training supervision fields:
  - scalar env: `candidate_tokens [S, A, E]`, `candidate_valid [S, A]`, `target_index [S]`, `belief_length` scalar
  - env batch: `candidate_tokens [B, S, A, E]`, `candidate_valid [B, S, A]`, `target_index [B, S]`, `belief_length [B]`
- For active rows in current variants (`A=2`), both opponent-owner candidates are valid and `target_index`
  identifies the true owner; inactive rows keep all candidates invalid and use `target_index = -1`.
- `belief_supervision()` returns dummy supervision (`belief_length = 0`, all candidates invalid,
  all `target_index = -1`) unless the state is an active-player first action-token stage.
  Chance/deal stages and intermediate multi-token action-prefix stages are dummy by contract.
- Consumers treat `belief_length > 0` as the supervised-root signal and must skip `belief_length = 0` states.
- Active rows point to canonical token triples `[PLAYER(x), RANK(r), HOLD]` through `target_index`;
  inactive rows use `target_index = -1` and no valid candidates.
- `E` is fixed by `MAX_BELIEF_TOKENS` (`3` in current DouDiZhu variants).
- `S` is fixed by `MAX_BELIEF_ROWS = 2 * BASE_HAND_CARDS + NUM_PUBLIC_LANDLORD_CARDS` (`37` in full DouDiZhu).
- `A` is fixed by `MAX_BELIEF_CANDIDATES` (`2` in current DouDiZhu variants).
- The target contains current remaining cards of the two opponents.
- If landlord is an opponent (peasant view), landlord public cards are included.
- Canonical ordering is rank high-to-low, then owner order `L -> P1 -> P2` within each rank (excluding self).
- `expand_belief_prefix_str*` expands one autoregressive belief step at a time and returns:
  - `candidate_tokens`: candidate next triples
  - `candidate_valid`: candidate validity mask
  - `candidate_belief_strs`: next canonical belief-string prefixes aligned with `candidate_tokens`
- Multiple candidates may be valid in a row; callers should keep branching behavior (no first-valid collapse).
- Candidate axis uses `MAX_BELIEF_CANDIDATES` (`2` in current DouDiZhu variants); do not hardcode literal `2`.
- Any branch flagged `candidate_valid` is guaranteed to be decodable as the next `belief_str` prefix.
- `determinize_history_str*` converts one complete belief trajectory into canonical `history_str` and rejects incomplete/non-prefix trajectories.
- `infoset_str_from_history_str*` derives only active-player infosets and fails fast on terminal/non-active contexts.
- `get_available_action_strs()` exposes live semantic action strings aligned with the current legal rows, which is useful
  for Kagami semantic trace recording during env-native rollout.

### History boundary contract (`decode` vs `encode`)

- `parse_infoset_str*` returns `History(boundary = ENDS_WITH_STATE)` with observation-aligned `player_ids`,
  `last_available_actions`, and `last_available_action_strs`.
- In ENDS_WITH_STATE, `action_taken_tokens[-1]` is PAD-only (`s,a,s,...,s`).
- `format_infoset_str` / `format_history_str` accept either boundary; state-ended histories are normalized by `EndpointFactory`.
- In ENDS_WITH_ACTION, `last_avail_*` tensors are absent; `action_taken_tokens[-1]` may be PAD-only for chance-prefix trajectories.
- `reconstruct_env(history_str, pending_action = "", seed = 0)` reconstructs one live environment state from canonical history.
- `reconstruct_envs(history_strs, pending_actions = [], seed = 0)` is the batched reconstruction helper.

Example conversion for round-trip encode:

```python
import nanopokerlib as npl

game = npl.doudizhu
decoded = game.parse_infoset_str("L+33445566778899T|L+BR2")
infoset_str = game.format_infoset_str(decoded)
history_str = game.format_history_str(decoded)
```

### Canonical history reconstruction (`reconstruct_env`)

DouDiZhu now supports reconstruction from canonical omniscient history strings:

```python
env, step = game.reconstruct_env(
    "L+<landlord_private>|P1+<peasant1_private>|P2+<peasant2_private>|L+<landlord_public>|..."
)
```

Contract details:

- The first 3 segments must be one private `+` segment per player (`L`, `P1`, `P2`), each with
  exactly `BASE_HAND_CARDS` cards.
- The next segment must be landlord public cards (`L+...`) with exactly `NUM_PUBLIC_LANDLORD_CARDS` cards.
- Remaining segments must be committed action suffix segments (`<Tag>-<move>`).
- Reconstruction is fail-fast for malformed segments, invalid turn order, and deck/card-count inconsistencies.
- `pending_action` uses the same semantic prefix grammar as `parse_infoset_str(...)`.

### Live canonical history advancement

DouDiZhu environments also expose:

- `Environment.advance_history_str(history_str: str) -> str`
- `EnvBatch.advance_history_strs(history_strs: list[str]) -> list[str]`

Contract details:

- Start from `""` after `reset()`, then consume exactly one `reset()` / `step()` transition per call.
- Private dealing, landlord-public reveal steps, and non-commit multi-token action-prefix steps return the same history unchanged.
- The opening root history appears on the transition from dealing into the landlord's first action.
- Each committed move appends exactly one canonical suffix segment.
- After a terminal auto-reset, the final history remains consumable once before the next advancement.

### Player tags

Segments are tagged with one of:

- `L` / `LANDLORD` → player `0`
- `P1` / `PEASANT1` → player `1`
- `P2` / `PEASANT2` → player `2`

Tags are case-insensitive and may contain surrounding whitespace.

### Format

`infoset_str` is a `|`-separated list of segments:

```
<ViewTag>+<private_cards> | L+<public_cards> | <Tag>-<move> | <Tag>-<move> | ...
```

#### Hand segments: `+`

Two `+` segments are used:

- `<ViewTag>+<private_cards>` declares the viewing player's **initial private hand**.
  - For `P1` / `P2`, this is the full initial hand (17 cards in `doudizhu`).
  - For `L`, this must exclude the 3 public landlord cards (so it is 17 cards in `doudizhu`).
- `L+<public_cards>` declares the 3 **public landlord cards** (broadcast to all players).

The first `+` segment determines the `view_player_id` and is also used when deterministically filling
unspecified information (see "Filling missing cards" below).

Examples:

- Landlord view: `L+334455...22|L+BR2|...`
- Peasant view: `P1+QKA2B|L+BR2|...`

#### Action segments: `-`

`<Tag>-<move>` records a **completed committed move** (a full card discard), not token-level action prefixes.

`<move>` is either:

- `PASS` (or empty), or
- a rank string of all cards discarded in that move (e.g. `777`, `BR`, `TTJJQQ`).

Examples:

- `P2-PASS`
- `L-3334`
- `P1-BR`

The action segments must be in valid turn order; reconstruction will throw if the history does not
match the game’s legal move order.

When using `parse_infoset_str` for action selection, the history should end on the viewing player's turn.
For example, we should expect `L+...|L+...|L-44|P1-PASS|P2-PASS`.

### Filling missing cards (important)

`parse_infoset_str` reconstructs a **single deterministic deal** consistent with the provided hand and move
history. It is not a belief sampler.

- The viewing player's private hand may be partial; missing private cards are filled deterministically (lowest
  ranks first) up to the expected private hand size.
- The landlord public cards segment must contain exactly 3 cards.
- Other players' hands must be omitted; they are seeded with the cards they played in the history and then
  deterministically filled to the expected hand size.

The input must still be **globally consistent** (e.g. no one can have more cards than expected, and the history
cannot spend more cards than exist in the deck).

### `pending_action`

`pending_action` is a **semantic action-prefix string** (e.g. `333+`, `333?`, `PASS`) that is applied after the
semantic `-` history is replayed.

After applying `pending_action`, `parse_infoset_str` returns an `History` whose final legal rows are exposed at
`infoset.last_available_actions`, and aligned semantic strings are exposed at
`infoset.last_available_action_strs`.

Prefixes use the following conventions:

- `?` = main group chosen, kicker type undecided (e.g. `333?`, `3333?`)
- `.` = one pending kicker card (e.g. `333.`, `3333..`, `333444....`)
- `+` = sequence start chosen, end not yet fixed (e.g. `3+`, `33+`, `333+`)

Example (Python, schematic):

```python
state = "L+...|L+...|L-44|P1-PASS|P2-PASS"
infoset = doudizhu.parse_infoset_str(state, "333+")
last = infoset.last_available_actions
action_strings = list(infoset.last_available_action_strs)
```
