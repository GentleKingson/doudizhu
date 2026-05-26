from __future__ import annotations

import tyro

from marvel.utils import configure_torch

from yomi.config import TrainConfig
from yomi.training.trainer import BeliefTrainer


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    configure_torch(cfg)
    BeliefTrainer(cfg).run()


if __name__ == "__main__":
    main()
