import argparse

import numpy as np
from stable_baselines3 import PPO

from affectively_environments.envs.heist import HeistEnvironment
from affectively_environments.envs.pirates import PiratesEnvironment
from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PPO model for Solid Rally Rally Single Objective RL.")
    parser.add_argument("--run", type=int, required=True, help="Run ID")
    parser.add_argument("--weight", type=float, required=True, help="Weight value for SORL reward")
    parser.add_argument("--game", required=True, help="Name of environment")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster index for Arousal Persona")
    parser.add_argument("--target_arousal", type=float, required=True, help="Target Arousal")
    parser.add_argument("--periodic_ra", type=int, required=True, help="Assign arousal rewards every 3 seconds, instead of synchronised with behavior.")
    args = parser.parse_args()

    run = args.run
    weight = args.weight
    env_type = args.game
    cluster = args.cluster
    period_ra = True if args.periodic_ra==1 else False

    if env_type.lower() == "pirates":
        env = PiratesEnvironment(id_number=run, weight=weight, graphics=True, logging=True, log_prefix="PPO/", period_ra=period_ra, cluster=cluster, targetArousal=args.target_arousal)
    elif env_type.lower() == "solid":
        env = SolidEnvironmentGameObs(id_number=run, weight=weight, graphics=True, logging=True,
                                      path="../Builds/MS_Solid/platform.exe", log_prefix="PPO/")
    elif env_type.lower() == "fps":
        env = HeistEnvironment(id_number=run, weight=weight, graphics=True, logging=True,
                               log_prefix="PPO/", path="../Builds/MS_Heist/Top-Down Shooter.exe")
    else:
        raise ValueError("Invalid environment type. Choose 'pirates' or 'solid'.")

    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    print(f"./Tensorboard/{PPO}/PPO-Metrics-{env_type.lower()}")
    model = PPO("MlpPolicy", env=env, tensorboard_log=f"../../Tensorboard/PPO/PPO-Metrics-{env_type.lower()}", device='cpu')
    model.learn(total_timesteps=10000000, progress_bar=True)
    model.save(f"./Agents/PPO/ppo_{env_type.lower()}_{label}_{run}")
