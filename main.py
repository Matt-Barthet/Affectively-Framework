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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PPO model for Solid Rally Rally Single Objective RL.")
    parser.add_argument("--run", type=int, required=True, help="Run ID")
    parser.add_argument("--weight", type=float, required=True, help="Weight value for SORL reward")
    parser.add_argument("--game", required=True, help="Name of environment")
    parser.add_argument("--target_arousal", type=float, required=True, help="Target Arousal")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster index for Arousal Persona")
    parser.add_argument("--periodic_ra", type=int, required=True, help="Assign arousal rewards every 3 seconds, instead of synchronised with behavior.")
    parser.add_argument("--cv", required=True, type=int, default=0)
    args = parser.parse_args()

    if args.cv == 0:
        if args.game == "fps":
            env = HeistEnvironmentGameObs(
                id_number=args.run,
                weight=args.weight,
                graphics=True,
                logging=True,
                log_prefix="PPO/",
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra
            )
        elif args.game == "solid":
            env = SolidEnvironmentGameObs(
                id_number=args.run,
                weight=args.weight,
                graphics=False,
                logging=True,
                log_prefix="PPO/",
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra
            )
        elif args.game == "platform":
            env = PiratesEnvironmentGameObs(
                id_number=args.run,
                weight=args.weight,
                graphics=True,
                logging=True,
                log_prefix="PPO/",
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra
            )
        model = PPO(
            policy="MlpPolicy",
            device='cpu',
            env=env,
        )
    elif args.cv == 1:
        # Note that CV builds cannot run in headless mode - the unity renderer must be switched on to produce frames.
        if args.game == "fps":
            env = HeistEnvironmentCV(
                id_number=args.run,
                weight=args.weight,
                logging=True,
                log_prefix="PPO/",
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra
            )
        elif args.game == "solid":
            env = SolidEnvironmentCV(
                id_number=args.run,
                weight=args.weight,
                logging=True,
                log_prefix="PPO/",
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra
            )
        elif args.game == "platform":
            env = PiratesEnvironmentCV(
                id_number=args.run,
                weight=args.weight,
                logging=True,
                log_prefix="PPO/",
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra
            )
        model = PPO(policy="CnnPolicy", env = env)

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
    model.save(f"./agents/PPO/cnn_ppo_solid_{label}_{args.run}_extended")