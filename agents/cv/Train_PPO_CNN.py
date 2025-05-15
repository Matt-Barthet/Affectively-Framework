from stable_baselines3 import PPO
from affectively_environments.envs.solid_cv import SolidEnvironmentCV
from stable_baselines3.common.callbacks import ProgressBarCallback

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from torchvision.models import resnet18, ResNet18_Weights


class CustomResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super(CustomResNetExtractor, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]  # Stacked frames count
        height, width = observation_space.shape[0], observation_space.shape[1]

        # Load a pre-defined ResNet and adapt it to the input shape
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=n_input_channels,  # Use stacked frames as input channels
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

        # Compute the output dimension of ResNet
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, height, width)
            sample_output = self.resnet(sample_input)
            resnet_output_dim = sample_output.shape[1]

        # Add a fully connected layer to map ResNet features to desired dimension
        self.linear = nn.Sequential(
            nn.Linear(resnet_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.permute(0, 3, 1, 2)
        resnet_output = self.resnet(observations)
        return self.linear(resnet_output)


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
        features_extractor_class=CustomResNetExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch = dict(pi=[256, 256], vf=[256, 256]),
        activation_fn = torch.nn.ReLU,
    )


    model = PPO(
        policy="CnnPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        tensorboard_log="./Tensorboard/CNN/",
        device='cuda',
    )
    model.learn(total_timesteps=10_000_000, callback=callbacks)
    model.save(f"./Agents/PPO/cnn_ppo_solid_{label}_{run}_extended")
