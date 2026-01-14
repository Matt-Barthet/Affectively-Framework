import os
import numpy as np

from GANEnv import GANLevelEnv
from stable_baselines3 import PPO
from GANGenerate import CNet

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch
import torch.nn as nn

def create_new_segment_folder(base_path="GeneratedLevelSegments"):
    os.makedirs(base_path, exist_ok=True)

    existing = [
        f for f in os.listdir(base_path)
        if f.startswith("SegmentsV") and f[9:].isdigit()
    ]

    if not existing:
        next_num = 1
    else:
        nums = [int(f[9:]) for f in existing]
        next_num = max(nums) + 1

    new_folder = os.path.join(base_path, f"SegmentsV{next_num}")
    os.makedirs(new_folder)

    return new_folder

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
    
def generate_graph_using_module():
    env = GANLevelEnv()
    # model = PPO.load("GANArousalAgents/MaxEnemy/cnn_ppo_solid_optimize_1_extended.zip", env=env)
    # model = PPO.load("GANArousalAgents/Playable/cnn_ppo_solid_onlyplayable_optimize_1_extended.zip", env=env)
    # obs, info = env.reset()

    episode_nums = []
    avg_enemy_counts = []

    episode_num = 100
    while episode_num <= 700:
        print("###################################")
        print("###################################")
        print("###################################")
        print("###################################")
        print("episode_num: " + str(episode_num))
        # model = PPO.load("GANArousalAgents/MaxEnemy/cnn_ppo_optimize_1_" + str(episode_num) + "_steps", env=env)
        model = PPO.load("GANArousalAgents/PPO/cnn_ppo_optimize_1_" + str(episode_num) + "_steps", env=env)
        average_num_enemy = 0
        for j in  range(10):
            print("ATTEMPT " + str(j))
            obs, info = env.reset()
            num_enemies = 0

            terminated = False
            for i in range(11):
                print("SEGMENT " + str(i))
                if not terminated:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)

                    full_level = info["full_level"]
                    num_enemies = sum(np.count_nonzero((arr == 5)) for arr in full_level)

            episode_nums.append(episode_num)
            avg_enemy_counts.append(num_enemies)

            average_num_enemy += num_enemies
            print(str(j) + ": " + str(num_enemies))

        average_num_enemy = average_num_enemy / 10
        # episode_nums.append(episode_num)
        # avg_enemy_counts.append(average_num_enemy)

        print("")
        print(str(average_num_enemy))
        print("====================")

        # if episode_num == 100:
        #     episode_num += 900
        # else:
        #     episode_num += 500
        episode_num += 300

    plt.figure()
    plt.plot(episode_nums, avg_enemy_counts, marker='o')
    plt.xlabel("Training Steps")
    plt.ylabel("Average Number of Enemies")
    plt.title("Enemy Count vs Training Progress")
    plt.grid(True)
    plt.show()
    
def generate_multiple_enemies_graph():
    log_file = "RecordedLogs\MultipleValuesRecordedWithoutPlayability\enemy_count.txt"

    values_1 = []
    values_2 = []
    values_3 = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(":")]
            if len(parts) != 3:
                continue

            v1, v2, v3 = map(float, parts)

            values_1.append(v1)
            values_2.append(v2)
            values_3.append(v3)

    x = range(len(values_1))

    plt.figure()
    plt.plot(x, values_1, label="Value 1")
    plt.plot(x, values_2, label="Value 2")
    plt.plot(x, values_3, label="Value 3")

    plt.xlabel("Level")
    plt.ylabel("Value")
    plt.title("Playable Results Breakdown")
    plt.legend()
    plt.show()

def generate_graph():
    # Read values from file
    # log_file = "RecordedLogs\MaxEnemyV2Logs\scores.txt"
    log_file = "RecordedLogs\MaxEnemyV2Logs\playable_results - Copy.txt"
    values = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                values.append(float(line))

    # X-axis: index (step, episode, etc.)
    x = range(len(values))

    # Plot
    plt.figure()
    plt.plot(x, values)
    plt.xlabel("Level")
    plt.ylabel("Number of Segments")
    plt.title("Number of Segments Before Being Unplayable (12 segments means 11 fully playable segments)")
    plt.show()

def generate_action_value_graph():
    env = GANLevelEnv()

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

    # model = PPO.load("GANArousalAgents/MaxEnemyV2/cnn_ppo_optimize_1_extended.zip", env=env)

    obs, info = env.reset()
    
    all_actions = []
    for i in range(11):
        print("i number: " + str(i))

        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        all_actions.append(action)

    all_actions = np.array(all_actions)
    timesteps = np.arange(all_actions.shape[0])

    plt.figure()

    # Plot each action dimension separately
    for dim in range(all_actions.shape[1]):
        plt.scatter(
            timesteps,
            all_actions[:, dim],
            label=f"Action dim {dim}",
            alpha=0.7,
        )

    plt.xlabel("Timestep")
    plt.ylabel("Action Value")
    plt.title("Scatter Plot of PPO Actions Over Time (Using New Untrained Model with Deterministic = False)")
    plt.legend()
    plt.show()

def unity_generate_level():
    results_path = create_new_segment_folder()

    env = GANLevelEnv(worker_id=0)

    # policy_kwargs = dict(
    #     features_extractor_class=CustomMLPExtractor,
    #     features_extractor_kwargs=dict(features_dim=256),
    #     net_arch = dict(pi=[256, 256], vf=[256, 256]),
    #     activation_fn = torch.nn.ReLU,
    # )
    
    # model = PPO(
    #     policy="MlpPolicy",
    #     policy_kwargs=policy_kwargs,
    #     env=env,
    #     verbose=1,
    #     n_steps=64,
    #     tensorboard_log="./Tensorboard/CNN/",
    #     device='cuda',
    # )

    model = PPO.load("GANArousalAgents/MultipleValuesWithoutPlayability", env=env)
    # GANArousalAgents\multipleenemieswithoutplayabilityv2.zip

    # model = PPO.load("GANArousalAgents/MaxEnemyV2/cnn_ppo_optimize_1_extended.zip", env=env)
    
    
    # model = PPO.load("GANArousalAgents/MaxEnemyV2/cnn_ppo_optimize_1_100_steps.zip", env=env)

    obs, info = env.reset()
    
    for i in range(11):
        print("i number: " + str(i))
        # action = np.random.uniform(-1, 1, size=32)
        # action = np.zeros(32) 
        # print("Action " + str(i) + ": " + str(action))

        # print("obs: " + str(obs))
        action, _ = model.predict(obs, deterministic=False)
        # obs, reward, terminated, truncated, info = env.step(i, action)
        obs, reward, terminated, truncated, info = env.step(action)

        # print("Generated Level (info):")
        # print(info)
        # print("Reward:", reward)

        segment = info["segment"]         # numpy array

        # print("")
        # print("Segment " + str(i + 1) + ": " + str(segment))
        # print("====================")

        file_path = os.path.join(results_path, f"segment_{i+1 + 1}.csv")
        np.savetxt(file_path, segment, fmt="%d", delimiter=",")

if __name__ == '__main__':
    generate_action_value_graph()