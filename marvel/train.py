# -*- coding: utf-8 -*-
import torch
import tyro

from marvel.config import TrainConfig
from marvel.training import registry
from marvel.utils import configure_torch

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    configure_torch(cfg)
    try:
        trainer_cls = registry.resolve(cfg.algo_name)
    except KeyError as exc:
        available = ", ".join(registry.available())
        raise ValueError(
            f"Unknown algorithm preset '{cfg.algo_name}'. "
            f"Available options: {available or 'none'}."
        ) from exc
    trainer_cls(cfg).run()


if __name__ == "__main__":
    main()
