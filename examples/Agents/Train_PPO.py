from stable_baselines3 import PPO

from affectively_environments.envs.heist import HeistEnvironment

import numpy as np

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)

    run = 1
    preference_task = True
    classification_task = False
    weight = 0.5

    env = HeistEnvironment(id_number=run, weight=weight, graphics=True, logging=True)
    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    model = PPO("MlpPolicy", env=env, tensorboard_log="../Tensorboard", device='cpu')
    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save(f"ppo_solid_{label}_{run}")
