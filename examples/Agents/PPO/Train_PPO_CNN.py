from stable_baselines3 import PPO
import numpy as np
from affectively_environments.envs.solid_cv import SolidEnvironmentCV
from stable_baselines3.common.callbacks import ProgressBarCallback
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch


class Custom3DCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super(Custom3DCNN, self).__init__(observation_space, features_dim)

        n_input_channels = 1  # Grayscale images
        stack_size = observation_space.shape[2]
        height, width = observation_space.shape[0], observation_space.shape[1]

        self.cnn = nn.Sequential(
            nn.Conv3d(
                in_channels=n_input_channels,
                out_channels=32,
                kernel_size=(3, 8, 8),
                stride=(1, 4, 4),
                padding=(1, 2, 2)
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Flatten()
        )

        # Compute the output dimension
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, stack_size, height, width)
            sample_output = self.cnn(sample_input)
            n_flatten = sample_output.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape observations to (batch_size, channels, depth, height, width)
        observations = observations.permute(0, 3, 1, 2).unsqueeze(1)
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)


if __name__ == "__main__":
    run = 1
    weight = 0

    env = SolidEnvironmentCV(
        id_number=run,
        weight=weight,
        graphics=True,
        logging=True,
        path="../Builds/MS_Solid/Racing.exe",
        log_prefix="CNN/",
        grayscale=False
    )

    label = 'optimize' if weight == 0 else 'arousal' if weight == 1 else 'blended'
    callbacks = ProgressBarCallback()

    policy_kwargs = dict(
        features_extractor_class=Custom3DCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        tensorboard_log="./Tensorboard/CNN/",
        device='cuda',

    )
    model.learn(total_timesteps=10_000_000, callback=callbacks)
    model.save(f"./Agents/PPO/cnn_ppo_solid_{label}_{run}_extended")
