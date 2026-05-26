from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_DATA = REPO_ROOT / "external" / "PerfectDou" / "eval_data_10k.pkl"


@dataclass(frozen = True)
class NetworkClientConfig:
    connect_timeout_s: float = 30.0
    request_timeout_s: float = 300.0
    max_batch_size: int = 256
    max_delay_ms: int = 2


@dataclass(frozen = True)
class NetworkServerConfig:
    socket_timeout_s: float = 300.0


@dataclass(frozen = True)
class DoudizhuHostConfig:
    num_games: int = 5000
    max_envs: int = 256
    progress_interval: int = 500
    ci_multiplier: float = 1.0
    perfect_info: bool = False

    def __post_init__(self) -> None:
        if float(self.ci_multiplier) != 1.0:
            raise ValueError("DoudizhuHostConfig.ci_multiplier must be exactly 1.0.")


@dataclass(frozen = True)
class PerfectDouAgentConfig:
    seed: int = 0


@dataclass(frozen = True)
class DouZeroAgentConfig:
    seed: int = 0


@dataclass(frozen = True)
class FPDouAgentConfig:
    eval_criteria: str = "winrate"
    checkpoint_dir: Path = field(default_factory = lambda: REPO_ROOT / "external" / "fpdou" / "checkpoints" / "final")
    out_conv: int = 64
    out_res: int = 64
    blocks: int = 5


@dataclass(frozen = True)
class MarvelSearchConfig:
    routine: str = "local-subgame"
    rollout_budget: int = 512
    search_floor: int | None = 16
    search_beta: float = 1.0
    seed: int = 0


@dataclass(frozen = True)
class MarvelAgentConfig:
    device: str = "cuda"
    action_temp: float = 0.0
    belief_temp: float = 1.0
    search_temp: float = 1.0
    search: MarvelSearchConfig = field(default_factory = MarvelSearchConfig)


DEFAULT_NETWORK_CLIENT = NetworkClientConfig()
DEFAULT_NETWORK_SERVER = NetworkServerConfig()
DEFAULT_HOST = DoudizhuHostConfig()
DEFAULT_PERFECTDOU_AGENT = PerfectDouAgentConfig()
DEFAULT_DOUZERO_AGENT = DouZeroAgentConfig()
DEFAULT_FPDOU_AGENT = FPDouAgentConfig()
DEFAULT_MARVEL_AGENT = MarvelAgentConfig()


__all__ = [
    "DEFAULT_DOUZERO_AGENT",
    "DEFAULT_EVAL_DATA",
    "DEFAULT_FPDOU_AGENT",
    "DEFAULT_HOST",
    "DEFAULT_MARVEL_AGENT",
    "DEFAULT_NETWORK_CLIENT",
    "DEFAULT_NETWORK_SERVER",
    "DEFAULT_PERFECTDOU_AGENT",
    "DouZeroAgentConfig",
    "FPDouAgentConfig",
    "DoudizhuHostConfig",
    "MarvelAgentConfig",
    "MarvelSearchConfig",
    "NetworkClientConfig",
    "NetworkServerConfig",
    "PerfectDouAgentConfig",
    "REPO_ROOT",
]
