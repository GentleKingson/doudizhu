from types import SimpleNamespace

import nanopokerlib as npl
import numpy as np
import pytest
import torch

from kagami.search.critic_value_estimator import CriticActionValueEstimator


ROOT = "L+33567888TTJQKKKA2|P1+4556677789TTKA22B|P2+3344456999JJJQQA2|L+QAR"


@pytest.mark.parametrize("suffix,pending", [("", ""), ("", "888?"), ("", "888."), ("", "KKK?"), ("", "KKK."), ("|L-5888|P1-PASS|P2-3JJJ", "")])
def test_critic_rollout_preserves_every_sampled_players_observations(suffix: str, pending: str) -> None:
    game = npl.doudizhu
    heads = ROOT.split("|")
    swapped = "|".join([heads[0], "P1+" + heads[2].partition("+")[2], "P2+" + heads[1].partition("+")[2], heads[3]])
    histories = [ROOT + suffix] if suffix else [ROOT, swapped]
    estimator = CriticActionValueEstimator(
        game = game, belief_model = None, critic_model = SimpleNamespace(output_mode = "qvalue", eval = lambda: None),
        device = torch.device("cpu"), belief_temp = 1.0, belief_sample_count = len(histories),
    )
    rollout, num_steps = estimator._build_critic_rollout(history_rows = histories, pending_action = pending, seed = 0)
    _, live = game.reconstruct_envs(histories, pending_actions = [pending] * len(histories), seed = 0)
    rows = np.arange(len(histories))
    np.testing.assert_array_equal(rollout["obs_tokens"][rows, num_steps - 1].numpy(), live.observations.tokens)
    np.testing.assert_array_equal(rollout["obs_channels"][rows, num_steps - 1].numpy(), live.observations.channels)
    _, root = game.reconstruct_envs([ROOT], seed = 0)
    root_step = int(game.parse_infoset_strs([game.infoset_str_from_history_str(ROOT)], [""]).num_steps[0]) - 1
    np.testing.assert_array_equal(rollout["obs_tokens"][0, root_step].numpy(), root.observations.tokens[0])
    np.testing.assert_array_equal(rollout["obs_channels"][0, root_step].numpy(), root.observations.channels[0])


def test_critic_rejects_runtime_without_full_history_parser() -> None:
    estimator = CriticActionValueEstimator(
        game = SimpleNamespace(), belief_model = None,
        critic_model = SimpleNamespace(output_mode = "qvalue", eval = lambda: None),
        device = torch.device("cpu"), belief_temp = 1.0, belief_sample_count = 1,
    )
    with pytest.raises(RuntimeError, match = "rebuild nanopokerlib"):
        estimator._build_critic_rollout(history_rows = [ROOT], pending_action = "", seed = 0)
