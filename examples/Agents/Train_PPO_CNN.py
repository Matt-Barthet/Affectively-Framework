from stable_baselines3 import PPO

from affectively_environments.envs.heist import HeistEnvironment
from affectively_environments.envs.solid import SolidEnvironment

import numpy as np

from affectively_environments.envs.solid_cv import SolidEnvironmentCV

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)

    run = 10
    preference_task = True
    classification_task = False
    weight = 0

    env = SolidEnvironmentCV(id_number=run, weight=weight, graphics=True, logging=True, path="../Builds/MS_Solid/Racing.exe",)
    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    model = PPO("CnnPolicy", env=env, tensorboard_log="./Tensorboard", device='cuda')
    model.learn(total_timesteps=10000000, progress_bar=True)
    model.save(f"ppo_solid_{label}_{run}")
