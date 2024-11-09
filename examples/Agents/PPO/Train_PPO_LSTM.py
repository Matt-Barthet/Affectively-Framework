from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import numpy as np
from affectively_environments.envs.solid_cv import SolidEnvironmentCV
from stable_baselines3.common.callbacks import ProgressBarCallback
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch

if __name__ == "__main__":
    run = 1
    weight = 0

    env = SolidEnvironmentCV(
        id_number=run,
        weight=weight,
        graphics=True,
        logging=True,
        path="../Builds/MS_Solid/Racing.exe",
        log_prefix="LSTM/",
        grayscale=False
    )

    label = 'optimize' if weight == 0 else 'arousal' if weight == 1 else 'blended'
    callbacks = ProgressBarCallback()

    model = RecurrentPPO(
        policy="CnnLstmPolicy",
        env=env,
        tensorboard_log="./Tensorboard/LSTM/",
        device='cuda',
    )
    model.learn(total_timesteps=10_000_000, callback=callbacks)
    model.save(f"./Agents/PPO/cnn_ppo_solid_{label}_{run}_extended")