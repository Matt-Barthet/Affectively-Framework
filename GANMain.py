import os
import numpy as np

from GANEnv import GANLevelEnv
from stable_baselines3 import PPO
from GANGenerate import CNet

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

if __name__ == '__main__':
    results_path = create_new_segment_folder()

    env = GANLevelEnv()
    model = PPO.load("GANArousalAgents/PPO/cnn_ppo_solid_optimize_1_extended.zip", env=env)
    obs, info = env.reset()

    for i in range(11):
        # action = np.random.uniform(-1, 1, size=32)
        # action = np.zeros(32) 
        # print("Action " + str(i) + ": " + str(action))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # print("Generated Level (info):")
        # print(info)
        # print("Reward:", reward)

        segment = info["segment"]         # numpy array

        print("")
        print("Segment " + str(i) + ": " + str(segment))
        print("====================")

        file_path = os.path.join(results_path, f"segment_{i+1}.csv")
        np.savetxt(file_path, segment, fmt="%d", delimiter=",")