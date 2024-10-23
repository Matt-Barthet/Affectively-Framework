from stable_baselines3 import PPO
import numpy as np
from affectively_environments.envs.solid_cv import SolidEnvironmentCV
from stable_baselines3.common.callbacks import ProgressBarCallback

if __name__ == "__main__":

    run = 1
    preference_task = True
    classification_task = False
    weight = 0

    env = SolidEnvironmentCV(id_number=run, weight=weight, graphics=True, logging=True, path="../Builds/MS_Solid/Racing.exe", log_prefix="CNN-")
    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    callbacks = ProgressBarCallback()

    model = PPO("CnnPolicy", env=env, tensorboard_log="./Tensorboard/CNN/", device='cuda', )
    model.learn(total_timesteps=10000000, callback=callbacks)
    model.save(f"./Agents/PPO/cnn_ppo_solid_{label}_{run}_extended")
