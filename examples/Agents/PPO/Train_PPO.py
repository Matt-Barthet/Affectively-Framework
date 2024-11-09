from stable_baselines3 import PPO

import numpy as np

from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from affectively_environments.envs.pirates import PiratesEnvironment

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)

    run = 5
    preference_task = True
    classification_task = False
    weight = 0

    env = PiratesEnvironment(id_number=run, weight=weight, graphics=True, logging=True,
                                  path="../Builds/MS_Pirates/platform.exe", log_prefix="PPO/")
    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    model = PPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard/", device='cpu')
    model.learn(total_timesteps=10000000, progress_bar=True)
    model.save(f"./Agents/PPO/ppo_pirates_{label}_{run}")
