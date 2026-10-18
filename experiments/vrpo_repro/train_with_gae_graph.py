"""Opt-in single-GPU VRPO entry point using the normal marvel.train arguments."""
from marvel.train import main as train_main
from marvel.training import vrpo
from experiments.vrpo_repro.gae_cuda_graph import GaeGraph


def main():
    original = vrpo.compute_gae_from_qvalue
    accelerator = GaeGraph(original)
    vrpo.compute_gae_from_qvalue = accelerator
    try:
        train_main()
    finally:
        vrpo.compute_gae_from_qvalue = original
        print('GAE CUDA Graph:', accelerator.stats(), flush=True)


if __name__ == '__main__':
    main()
