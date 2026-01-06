from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from torchvision.models import resnet18, ResNet18_Weights

import numpy as np

from GANEnv import GANLevelEnv
from GANGenerate import CNet

import os
import shutil
import platform
import sys
import yaml
import subprocess
import sys
# print("SHOWING")
# print(sys.executable)

class CustomMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input = np.prod(observation_space.shape)
        self.linear = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    def forward(self, obs):
        return self.linear(obs.flatten(start_dim=1))

def load_config(config_file_path):
    """
    Loads parameters from a YAML configuration file.

    Args:
        config_file_path (str): The full path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration parameters.
    """
    try:
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file '{config_file_path}': {exc}")
        return None

if __name__ == "__main__":    
    run = 1
    weight = 0

    env = GANLevelEnv()
    env = Monitor(env)

    label = 'optimize' if weight == 0 else 'arousal' if weight == 1 else 'blended'

    checkpoint_callback = CheckpointCallback(
                                                save_freq=100,
                                                save_path="./GANArousalAgents/PPO/",
                                                name_prefix=f"cnn_ppo_onlyplayable_{label}_{run}"
                                            )
    
    callbacks = CallbackList([
                                ProgressBarCallback(),
                                checkpoint_callback
                            ])


    policy_kwargs = dict(
        features_extractor_class=CustomMLPExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch = dict(pi=[256, 256], vf=[256, 256]),
        activation_fn = torch.nn.ReLU,
    )

    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        verbose=1,
        n_steps=64,
        tensorboard_log="./Tensorboard/CNN/",
        device='cuda',
    )

    model.learn(total_timesteps=4000, callback=callbacks)
    model.save(f"./GANArousalAgents/PPO/cnn_ppo_solid_onlyplayable_{label}_{run}_extended")