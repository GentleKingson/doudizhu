"""Small real-native VRPO fork probe, deliberately separate from the 12 formal runs."""
import argparse
import json
import torch
from pathlib import Path
from marvel.config import TrainConfig, ModelConfig, PPOConfig, VRPOConfig
from marvel.runtime.full_state import full_config, write_json
from marvel.training.vrpo import VRPOTrainer
from marvel.training.workflow import TrainingWorkflow
from marvel.utils import configure_torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--total', type=int, default=6)
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    torch.set_num_interop_threads(1)
    if args.resume:
        cfg = full_config(json.loads((Path(args.resume) / 'config.json').read_text()))
        cfg.resume_from = args.resume
        cfg.total_iterations = args.total
        cfg.logging_dir = args.output
        cfg.run_name = Path(args.output).name
    else:
        cfg = TrainConfig(device=args.device, seed=17, use_compile=False, deterministic=not args.fast,
            deterministic_disable_tf32=not args.fast, full_state_checkpoints=True,
            total_iterations=args.total, checkpoint_interval=3, evaluation_interval=-1, param_stats_interval=-1,
            rollout_batch_size=4, collect_batch_size=4, mini_batches=2, replay_buffer_scale=2,
            lr_stable_iterations=5, reg_stable_iterations=2, agent_merging_beta=.999,
            model=ModelConfig(history_dim=8, feature_dim=16, decoder_layers=1, decoder_heads=2,
                              state_layers=1, action_layers=1, fuse_layers=1),
            ppo=PPOConfig(agent_epoch=1), vrpo=VRPOConfig(aux_epoch=2, replay_buffer_device='cpu'),
            logging_dir=args.output)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True
    configure_torch(cfg)
    def factory(config):
        runtime = TrainingWorkflow(config, game=config.game)
        save = runtime.checkpoint_policy.save

        def save_with_leftover(state):
            if state.iteration == 3:
                # Both forks apply this same probe-only intervention using real native transitions.
                sampler = runtime.sampler
                sampler.prepare_agent_cache(agent_model=state['agent_resume_state_dict'])
                with torch.inference_mode():
                    while sampler.commit_buffer_size == 0:
                        sampler.rollout_step(agent_model=state['agent_resume_state_dict'])
                runtime.sync_step_totals_from_sampler()
                state['total_env_steps'] = sampler.total_env_steps
                state['total_game_steps'] = sampler.total_game_steps
            save(state)

        runtime.checkpoint_policy.save = save_with_leftover
        return runtime

    VRPOTrainer(cfg, runtime_factory=factory).run()
    generations = sorted((Path(args.output) / 'checkpoints').glob('i*-g*'))
    write_json(Path(args.output) / 'probe_result.json', {'generation':str(generations[-1].resolve())})


if __name__ == '__main__':
    main()
