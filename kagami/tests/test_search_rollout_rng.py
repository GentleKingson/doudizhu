from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kagami.search.rollout_session import SearchRolloutSession


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_search_rollout_rng_is_local_and_reproducible(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def run(seed):
        class SamplingSampler:
            parsed_envs_step = SimpleNamespace(
                batch_size = torch.Size([32]),
                outcome = {
                    "is_terminal": torch.zeros(32, dtype = torch.bool),
                    "rewards": torch.zeros((32, 3), dtype = torch.float64),
                },
            )
            calls = 0

            def prepare_agent_cache(self, *, agent_model):
                pass

            def rollout_step(self, *, agent_model, generator, forced_action_indices = None):
                self.calls += 1
                choices = torch.multinomial(
                    torch.ones((32, 8), device = device), 1, generator = generator,
                ).squeeze(1).cpu()
                self.parsed_envs_step.outcome["rewards"][:, 0] += choices * (10 ** self.calls)
                return torch.full((32,), self.calls == 2, dtype = torch.bool)

        before = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
        result = SearchRolloutSession(
            game = object(), policy_model = object(), root_player_id = 0,
            rollout_temp = 1.0, max_horizon = 3,
        ).run_sampler_rollout(
            sampler = SamplingSampler(), agent_model = object(),
            forced_first_action_indices = torch.zeros(32, dtype = torch.long),
            generator = torch.Generator(device = device).manual_seed(seed),
        )
        after = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
        assert torch.equal(before, after)
        return result

    first = run(73)
    torch.rand(1000, device = device)
    assert np.array_equal(first, run(73))
    assert not np.array_equal(first, run(74))
