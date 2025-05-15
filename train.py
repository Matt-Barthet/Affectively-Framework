import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

from affectively.environments.pirates_cv import PiratesEnvironmentCV
from affectively.environments.heist_cv import HeistEnvironmentCV
from affectively.environments.solid_cv import SolidEnvironmentCV
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PPO model for Solid Rally Rally Single Objective RL.")
    parser.add_argument("--run", type=int, required=True, help="Run ID")
    parser.add_argument("--weight", type=float, required=True, help="Weight value for SORL reward")
    parser.add_argument("--game", required=True, help="Name of environment")
    parser.add_argument("--target_arousal", type=float, required=True, help="Target Arousal")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster index for Arousal Persona")
    parser.add_argument("--periodic_ra", type=int, required=True, help="Assign arousal rewards every 3 seconds, instead of synchronised with behavior.")
    parser.add_argument("--cv", required=True, type=int, help="0 for GameObs, 1 for CV")
    parser.add_argument("--headless", required=True, type=int, help="0 for headless mode, 1 for graphics mode")
    parser.add_argument("--logdir", required=True, help="Log directory for TensorBoard")
    parser.add_argument("--grayscale", required=True, type=int, help="0 for RGB, 1 for grayscale")
    parser.add_argument("--discretize", required=True, type=int, help="0 for continuous, 1 for discretized observations")
    parser.add_argument("--algorithm", required=True, help="Algorithm to use for training")
    parser.add_argument("--policy", required=True, help="Policy to use for training")
    args = parser.parse_args()

    if args.cv == 0:
        if args.game == "fps":
            env = HeistEnvironmentGameObs(
                id_number=args.run,
                weight=args.weight,
                graphics=args.headless==0,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                discretize=args.discretize
            )
        elif args.game == "solid":
            env = SolidEnvironmentGameObs(
                id_number=args.run,
                weight=args.weight,
                graphics=args.headless==0,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                discretize=args.discretize
            )
        elif args.game == "platform":
            env = PiratesEnvironmentGameObs(
                id_number=args.run,
                weight=args.weight,
                graphics=args.headless==0,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                discretize=args.discretize
            )
        model = PPO(policy=args.policy, device='cpu', env=env) # Define model that trains using game states here.
    elif args.cv == 1:  # CV builds cannot run in headless mode - the unity renderer must be switched on to produce frames.
        if args.game == "fps":
            env = HeistEnvironmentCV(
                id_number=args.run,
                weight=args.weight,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                grayscale=args.grayscale
            )
        elif args.game == "solid":
            env = SolidEnvironmentCV(
                id_number=args.run,
                weight=args.weight,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                grayscale=args.grayscale
            )
        elif args.game == "platform":
            env = PiratesEnvironmentCV(
                id_number=args.run,
                weight=args.weight,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                grayscale=args.grayscale
            )

        model = PPO(policy=args.policy, env = env) # define model for training using pixels here


    experiment_name = f'{args.logdir}/{args.game}/{"Maximize Arousal" if args.target_arousal == 1 else "Minimize Arousal"}/{args.algorithm}/{args.policy}-Cluster{args.cluster}-{args.weight}λ-run{args.run}'
    env.callback =  TensorBoardCallback(experiment_name, env)

    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # env = VecTransposeImage(env)  # Fix channel order

    # policy_kwargs = dict(
        # features_extractor_class=CustomResNetExtractor,
        # features_extractor_kwargs=dict(features_dim=256),
    #     net_arch = dict(pi=[256, 256], vf=[256, 256]),
    #     activation_fn = torch.nn.ReLU,
    #     lstm_hidden_size=256,
    #     n_lstm_layers=1,
    #     shared_lstm=True,
    #     enable_critic_lstm=False,
    #     normalize_images=False,
    # )

    label = 'optimize' if args.weight == 0 else 'arousal' if args.weight == 1 else 'blended'
    callbacks = ProgressBarCallback()

    model.learn(total_timesteps=5_000_000, callback=callbacks)
    model.save(f"{experiment_name}.zip")
