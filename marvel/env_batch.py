from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from marvel.model.tensordict import TensorDict

from marvel.config import TrainConfig

__all__ = [
    "EnvOutcome",
    "EnvStepBatch",
]


@dataclass(frozen = True)
class EnvOutcome:
    """Rewards and termination flags produced after an environment step."""

    data: TensorDict

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.data[key]


class EnvStepBatch:
    """Parsed environment response used as agent input and for buffer commits."""

    def __init__(self, observation: TensorDict, outcome: EnvOutcome):
        self._observation = observation
        self._outcome = outcome

    @classmethod
    def from_env_step(cls, envs_step, cfg: TrainConfig) -> "EnvStepBatch":
        """Convert a raw environment step into structured tensors."""
        rewards = envs_step.rewards
        batch = int(rewards.shape[0])
        device = torch.device(cfg.device)

        outcome = EnvOutcome(
            TensorDict(
                {
                    "rewards": rewards,
                    "is_terminal": envs_step.is_terminal.astype(bool),
                },
                batch_size = [batch],
                device = device,
            )
        )

        observations = envs_step.observations
        observation = TensorDict(
            {
                "obs_tokens": observations.tokens.astype(np.int64),
                # Channels are 0/1 indicators; store as uint8 to shrink buffer footprint.
                "obs_channels": observations.channels.astype(np.uint8),
                "player_ids": envs_step.player_ids.astype(np.int64),
                # Keep this lazy getter after observation/player conversion errors.
                "avail_tokens": (available_actions := envs_step.available_actions).tokens.astype(np.int64),
                "avail_weights": available_actions.weights,
                "avail_is_commit": available_actions.is_commit.astype(bool),
            },
            batch_size = [batch],
            device = device,
        )
        return cls(observation, outcome)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._observation[key]

    @property
    def device(self) -> torch.device:
        return self._observation.device

    @property
    def batch_size(self) -> torch.Size:
        return self._observation.batch_size

    @property
    def outcome(self) -> EnvOutcome:
        return self._outcome

    def record_agent_outputs(
        self,
        *,
        avail_probs: torch.Tensor | None = None,
        action_taken: torch.Tensor | None = None,
        step_fields: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Attach agent outputs for the current decision step."""
        if avail_probs is not None:
            self._observation.set("avail_probs", avail_probs)
        if action_taken is not None:
            self._observation.set("action_taken", action_taken)
        if step_fields:
            for name, tensor in step_fields.items():
                self._observation.set(name, tensor)

    def build_transition(self, outcome: EnvOutcome | None = None) -> TensorDict:
        """Combine observation, agent outputs, and outcome into a transition record."""
        final_outcome = outcome.data if outcome is not None else self._outcome.data

        merged = {
            key: value
            for key, value in self._observation.items()
        }
        merged.update({key: value for key, value in final_outcome.items()})
        return TensorDict(
            merged,
            batch_size = self._observation.batch_size,
            device = self._observation.device,
        )
